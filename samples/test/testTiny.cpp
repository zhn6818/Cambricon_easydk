#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/time.h>
#include <unistd.h>

#include <csignal>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cnpostproc.h"
#include "device/mlu_context.h"
#include "easybang/resize_and_colorcvt.h"
#include "easycodec/easy_decode.h"
#include "easyinfer/easy_infer.h"
#include "easyinfer/mlu_memory_op.h"
#include "easyinfer/model_loader.h"
#include "easytrack/easy_track.h"
#include "inference.h"

DEFINE_bool(show, false, "show image");
DEFINE_bool(save_video, true, "save output to local video file");
DEFINE_int32(repeat_time, 0, "process repeat time");
DEFINE_string(data_path, "", "video path");
// DEFINE_string(model_path, "/data1/wcl/mlu/174/zhn/yolo/suguan/yolov3_4batch4core_simple.cambricon", "infer offline
// model path");
// DEFINE_string(model_path, "/data1/zhn/suguan/yolov3_1b4c_simple.cambricon", "infer offline model path");
DEFINE_string(model_path, "/data1/zhn/yiliuwu/yiliuwu_yolov3_4b4c_simple_270.cambricon", "infer offline model path");
// DEFINE_string(model_path, "/data1/zhn/headAbandon/yiliuwu_1b4c_simple.cambricon", "infer offline model path");
DEFINE_string(label_path, "", "label path");
DEFINE_string(func_name, "subnet0", "model function name");
DEFINE_string(track_model_path, "", "track model path");
DEFINE_string(track_func_name, "subnet0", "track model function name");
DEFINE_int32(wait_time, 0, "time of one test case");
DEFINE_string(net_type, "", "neural network type, SSD or YOLOv3");

void preYolov3(cv::Mat& matIn, int iWidth, int iHeight, cv::Mat& out) {
  // if(out.empty()){
  //   out = cv::Mat(iHeight, iWidth, CV_8UC3, cv::Scalar::all(0));
  // }
  int w = iWidth;
  int h = iHeight;
  int c = matIn.channels();

  int imw = matIn.cols;
  int imh = matIn.rows;

  int new_w = imw;
  int new_h = imh;

  if (((float)w / imw) < ((float)h / imh)) {
    new_w = w;
    new_h = (imh * w) / imw;
  } else {
    new_h = h;
    new_w = (imw * h) / imh;
  }
  cv::Mat mat(new_h, new_w, CV_8UC3);
  cv::resize(matIn, mat, cv::Size(new_w, new_h));

  cv::Mat src(h, w, CV_8UC3, cv::Scalar::all(127));
  cv::Mat srcROI = src(cv::Rect((w - new_w) / 2, (h - new_h) / 2, new_w, new_h));
  mat.copyTo(srcROI);

  src.copyTo(out);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "FLAGS_func_name:" << FLAGS_func_name << std::endl;

  std::cout << "make_start " << std::endl;
  std::shared_ptr<Detection> g_runner = std::make_shared<Detection>(FLAGS_model_path, FLAGS_func_name, 0);
  std::cout << "make_end " << std::endl;
  std::vector<cv::Mat> batch_image;
  std::vector<cv::Size> vecSize;
  cv::Mat img1 = cv::imread("/data1/zhn/yiliuwu/images/03010007065000000.mp4_20210827_221438.628.jpg");
  int h = img1.rows, w = img1.cols;
  cv::Mat img2;
  cv::resize(img1, img2, cv::Size(w - 10, h));
  cv::Mat img3;
  cv::resize(img1, img3, cv::Size(w, h - 40));
  cv::Mat img4;
  cv::resize(img1, img4, cv::Size(w - 300, h - 500));
  // cv::Mat img2 = cv::imread("/data1/zhn/suguan/testImg/0_100_out.jpg");
  // cv::Mat img3 = cv::imread("/data1/zhn/suguan/testImg/0_150_out.jpg");
  // cv::Mat img4 = cv::imread("/data1/zhn/suguan/testImg/0_200_out.jpg");

  cv::Size sizeGrunner = g_runner->GetSize();

  cv::Mat img1_resize;
  preYolov3(img1, sizeGrunner.width, sizeGrunner.height, img1_resize);
  std::cout << "info : " << img1_resize.cols << " " << img1_resize.rows << std::endl;
  batch_image.push_back(img1_resize);
  vecSize.push_back(cv::Size(img1.cols, img1.rows));

  cv::Mat img2_resize;
  preYolov3(img2, sizeGrunner.width, sizeGrunner.height, img2_resize);
  std::cout << "info : " << img2_resize.cols << " " << img2_resize.rows << std::endl;
  batch_image.push_back(img2_resize);
  vecSize.push_back(cv::Size(img2.cols, img2.rows));

  cv::Mat img3_resize;
  preYolov3(img3, sizeGrunner.width, sizeGrunner.height, img3_resize);
  std::cout << "info : " << img3_resize.cols << " " << img3_resize.rows << std::endl;
  batch_image.push_back(img3_resize);
  vecSize.push_back(cv::Size(img3.cols, img3.rows));

  cv::Mat img4_resize;
  preYolov3(img4, sizeGrunner.width, sizeGrunner.height, img4_resize);
  std::cout << "info : " << img4_resize.cols << " " << img4_resize.rows << std::endl;
  batch_image.push_back(img4_resize);
  vecSize.push_back(cv::Size(img4.cols, img4.rows));

  std::vector<std::vector<DetectedObject>> arrDetection;
  g_runner->Detect(batch_image, arrDetection, vecSize);
  std::cout << "arrDetection size() " << arrDetection.size() << std::endl;

  // for (size_t nn = 0; nn < arrDetection.size(); nn++)
  // {
  std::string iStr = "/data1/zhn/yiliuwu/result/out_" + std::to_string(0) + ".png";
  for (size_t nn = 0; nn < arrDetection[0].size(); nn++) {
    cv::Rect rect = arrDetection[0][nn].bounding_box;
    cv::rectangle(img1, rect, cv::Scalar(0, 128, 255), 2);
    std::string txt = std::to_string(arrDetection[0][nn].object_class) + "_" + std::to_string(arrDetection[0][nn].prob);
    cv::putText(img1, txt, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 3, 8);
  }
  // }
  cv::imwrite(iStr, img1);

  iStr = "/data1/zhn/yiliuwu/result/out_" + std::to_string(1) + ".png";
  for (size_t nn = 0; nn < arrDetection[1].size(); nn++) {
    cv::Rect rect = arrDetection[1][nn].bounding_box;
    cv::rectangle(img2, rect, cv::Scalar(0, 128, 255), 2);
    std::string txt = std::to_string(arrDetection[1][nn].object_class) + "_" + std::to_string(arrDetection[1][nn].prob);
    cv::putText(img2, txt, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 3, 8);
  }
  // }
  cv::imwrite(iStr, img2);

  iStr = "/data1/zhn/yiliuwu/result/out_" + std::to_string(2) + ".png";
  for (size_t nn = 0; nn < arrDetection[2].size(); nn++) {
    cv::Rect rect = arrDetection[2][nn].bounding_box;
    cv::rectangle(img3, rect, cv::Scalar(0, 128, 255), 2);
    std::string txt = std::to_string(arrDetection[2][nn].object_class) + "_" + std::to_string(arrDetection[2][nn].prob);
    cv::putText(img3, txt, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 3, 8);
  }
  // }
  cv::imwrite(iStr, img3);

  iStr = "/data1/zhn/yiliuwu/result/out_" + std::to_string(3) + ".png";
  for (size_t nn = 0; nn < arrDetection[3].size(); nn++) {
    cv::Rect rect = arrDetection[3][nn].bounding_box;
    cv::rectangle(img4, rect, cv::Scalar(0, 128, 255), 2);
    std::string txt = std::to_string(arrDetection[3][nn].object_class) + "_" + std::to_string(arrDetection[3][nn].prob);
    cv::putText(img4, txt, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 3, 8);
  }
  // }
  cv::imwrite(iStr, img4);
  return 0;
}