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
using namespace cv;

DEFINE_bool(show, false, "show image");
DEFINE_bool(save_video, true, "save output to local video file");
DEFINE_int32(repeat_time, 0, "process repeat time");
DEFINE_string(data_path, "", "video path");
// DEFINE_string(model_path, "/data1/wcl/mlu/174/zhn/yolo/suguan/yolov3_4batch4core_simple.cambricon", "infer offline
// model path");
DEFINE_string(model_path, "/data/ld/project/yolov4-csp-mlu/yolov4_csp_int8_1_4.cambricon", "infer offline model path");
//DEFINE_string(model_path, "/data1/zhn/suguan/yolov3_4b4c_simple.cambricon", "infer offline model path");
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


int main()
{
    std::string videoName = "/data/ld/test_video/50005_parking2021-07-14-12-39-44.mp4";
    std::string savename = "/data/ld/test_video/res.avi";
    cv::VideoCapture cap(videoName);
    cv::Size videoSize(cap.get(CAP_PROP_FRAME_WIDTH),cap.get(CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer;
    writer.open(savename,VideoWriter::fourcc('D','I','V','X'),20, videoSize);
    cv::Mat frame;
    HDDetectYolov3 hdDetect(FLAGS_model_path,FLAGS_func_name);
    while(true)
    {
        if(!cap.read(frame))
            break;
        cv::Mat resizeImg;
        preYolov3(frame, hdDetect.GetSize().width, hdDetect.GetSize().height, resizeImg);
      
        std::vector<std::vector<DetectedObject>> res;
        std::vector<cv::Rect> rects;
        rects.push_back(cv::Rect(450,66,416,416));
        rects.push_back(cv::Rect(1271,44,416,416));
        MatAndRois matAndRois;
        matAndRois.input = resizeImg;
        matAndRois.sizeInput = cv::Size(frame.cols,frame.rows);
        for (auto& rect:rects)
        {
            cv::Mat roi = frame(rect).clone();
            cv::Mat resizeRoi ;
            preYolov3(roi,hdDetect.GetSize().width, hdDetect.GetSize().height,resizeRoi);
            matAndRois.matRois.push_back(resizeRoi);
        }
        matAndRois.rectRois = rects;
        std::vector<MatAndRois>matAndRoisVec{matAndRois};
        hdDetect.detect2(matAndRoisVec,res);
        for (size_t nn = 0; nn < res[0].size(); nn++) 
        {
            cv::Rect rect = res[0][nn].bounding_box;
            cv::rectangle(frame, rect, cv::Scalar(0, 128, 255), 2);
            std::string txt = std::to_string(res[0][nn].object_class) + "_" + std::to_string(res[0][nn].prob);
            cv::putText(frame, txt, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255),2);
        }
        for(auto& rect : rects)
        {
            cv::rectangle(frame,rect,cv::Scalar(0,0,255));
        }
        writer.write(frame);
    }
}