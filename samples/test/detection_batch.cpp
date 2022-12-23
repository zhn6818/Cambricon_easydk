#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/time.h>
#include <unistd.h>

#include <csignal>
#include <fstream>
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
DEFINE_string(data_path, "/data1/zhn/zg/images/test.txt", "video path");
// DEFINE_string(model_path, "/data1/wcl/mlu/174/zhn/yolo/suguan/yolov3_4batch4core_simple.cambricon", "infer offline
// model path");
// DEFINE_string(model_path, "/data1/zhn/suguan/yolov3_1b4c_simple.cambricon", "infer offline model path");
DEFINE_string(model_path, "/data1/zhn/zg/zg_yolov3_4b4c_simple_270.cambricon", "infer offline model path");
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
  std::cout << "FLAGS_data_path: " << FLAGS_data_path << std::endl;
  std::cout << "make_start " << std::endl;
  std::shared_ptr<Detection> g_runner = std::make_shared<Detection>(FLAGS_model_path, FLAGS_func_name, 0);
  std::cout << "make_end " << std::endl;

  // cv::Mat img = cv::imread("/data1/zhn/suguan/testImg/0_200.jpg");
  // cv::Mat resizeMat;
  // cv::resize(img, resizeMat, cv::Size(img.cols - 500, img.rows - 500));
  // cv::imwrite("/data1/zhn/suguan/testImg/0_200_out.jpg", resizeMat);

  std::vector<cv::Mat> batch_image;
  std::vector<cv::Size> vecSize;
  std::string line;
  int index = 0;
  std::fstream videoTxtFile(FLAGS_data_path);
  while (getline(videoTxtFile, line)) {
    std::cout << line << std::endl;
    cv::Mat img1 = cv::imread(line);
    cv::Size sizeGrunner = g_runner->GetSize();
    cv::Mat img1_resize;
    preYolov3(img1, sizeGrunner.width, sizeGrunner.height, img1_resize);
    std::vector<cv::Mat> batch_image;
    std::vector<cv::Size> vecSize;
    for (int i = 0; i < g_runner->GetBatch(); i++) {
      batch_image.push_back(img1_resize);
    }

    vecSize.push_back(cv::Size(img1.cols, img1.rows));
    std::vector<std::vector<DetectedObject>> arrDetection;
    g_runner->Detect(batch_image, arrDetection, vecSize);
    std::string iStr = "/data1/zhn/zg/res/out_" + std::to_string(index) + ".png";
    for (size_t nn = 0; nn < arrDetection[0].size(); nn++) {
      cv::Rect rect = arrDetection[0][nn].bounding_box;
      cv::rectangle(img1, rect, cv::Scalar(0, 128, 255), 2);
      std::string txt =
          std::to_string(arrDetection[0][nn].object_class) + "_" + std::to_string(arrDetection[0][nn].prob);
      cv::putText(img1, txt, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 3, 8);
    }
    // }
    cv::imwrite(iStr, img1);
    index += 1;
  }
  return 0;
}
