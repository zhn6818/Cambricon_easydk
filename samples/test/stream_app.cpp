/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

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
//DEFINE_string(model_path, "/data1/wcl/mlu/174/zhn/yolo/suguan/yolov3_4batch4core_simple.cambricon", "infer offline model path");
DEFINE_string(model_path, "/data1/zhn/suguan/yolov3_1b4c_simple.cambricon", "infer offline model path");
//DEFINE_string(model_path, "/data1/zhn/headAbandon/yiliuwu_1b4c_simple.cambricon", "infer offline model path");
DEFINE_string(label_path, "", "label path");
DEFINE_string(func_name, "subnet0", "model function name");
DEFINE_string(track_model_path, "", "track model path");
DEFINE_string(track_func_name, "subnet0", "track model function name");
DEFINE_int32(wait_time, 0, "time of one test case");
DEFINE_string(net_type, "", "neural network type, SSD or YOLOv3");

// std::shared_ptr<StreamRunner> g_runner;
// bool g_exit = false;

// void HandleSignal(int sig) {
//   g_runner->Stop();
//   g_exit = true;
//   LOG(INFO) << "Got INT signal, ready to exit!";
// }

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // check params
  // CHECK_NE(FLAGS_data_path.size(), 0u);
  // CHECK_NE(FLAGS_model_path.size(), 0u);
  // CHECK_NE(FLAGS_func_name.size(), 0u);
  // CHECK_NE(FLAGS_label_path.size(), 0u);
  // CHECK_NE(FLAGS_net_type.size(), 0u);
  // CHECK_GE(FLAGS_wait_time, 0);
  // CHECK_GE(FLAGS_repeat_time, 0);

  // FLAGS_data_path
  std::cout << "FLAGS_func_name:" << FLAGS_func_name << std::endl;
  // try {
  // while(1)
  // {}
  std::cout << "make_start " << std::endl;
  std::shared_ptr<DetectionRunner> g_runner = std::make_shared<DetectionRunner>(FLAGS_model_path, FLAGS_func_name, 0);
  std::cout << "make_end " << std::endl;
  // } catch (edk::Exception& e) {
  //   LOG(ERROR) << "Create stream runner failed" << e.what();
  //   return -1;
  // }
  std::string strpath = "/data1/wcl/mlu/174/wcl/data/suguan/video/50022_inverse2021-01-08-12-13-52.mp4";
  //std::string strpath = "/data1/zhn/headAbandon/videos/NLCJ_wan_passenger.asf";
  std::string savename = "/data1/zhn/out/out.avi";
  std::unique_ptr<cv::VideoWriter> video_writer_{nullptr};
  cv::VideoCapture capture(strpath);
  // cv::VideoCapture capture("rtsp://admin:admin@10.0.0.241:554/cam/realmonitor?channel=1&subtype=0");
  if (!capture.isOpened()) std::cout << "fail to open!" << std::endl;
  double rate = capture.get(cv::CAP_PROP_FPS);
  std::cout << "fps:" << rate << std::endl;
  long totalFrameNumber = capture.get(cv::CAP_PROP_FRAME_COUNT);
  std::cout << "total frames:" << totalFrameNumber << "frames" << std::endl;
  bool stop = false;
  cv::Mat frame, frame1;
  long currentFrame = 0;

  //static const cv::Size g_out_video_size = cv::Size(1280, 720);
  cv::Size g_out_video_size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  // writer.open(savename,CV_FOURCC('P','I','M','1'),rate, videoSize);
  video_writer_.reset(new cv::VideoWriter(savename, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), (int)rate, g_out_video_size));

  // std::string img_name = "/data/wcl/174/wcl/data/suguan/image/JPEGImages/0_950.jpg";
  // std::string savename = "/data/models/wcl/data/suguan/image/0_950_out.jpg";
  // cv::Mat image = cv::imread(img_name);
  // std::vector<DetectedObject> arrDetection;

  while (!stop) {
    if (!capture.read(frame)) {
      break;
    }
    currentFrame++;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~    currentFrame ~~~~~~~~~~~~~~~~~~~~~~~: " << currentFrame << std::endl;
    if (frame.empty()) break;

    frame1 = frame.clone();
    int batch = g_runner->GetBatch();
    std::vector<cv::Mat> batch_image;
    for(size_t i = 0; i < batch; i++)
    {
      batch_image.push_back(frame);
    }
    
    std::vector<std::vector<cv::Mat>> preprocessedImages;
    std::vector<cv::Size> image_paras;
    std::vector<std::vector<DetectedObject>> arrDetection;
    g_runner->Pre(batch_image, preprocessedImages, image_paras);
    g_runner->Detect(preprocessedImages, arrDetection, image_paras);

    std::cout << "arrDetection size() " << arrDetection.size() << std::endl;

    for(size_t nn = 0 ; nn < arrDetection[0].size(); nn++)
    {
      cv::Rect rect = arrDetection[0][nn].bounding_box;
      cv::rectangle(frame1, rect, cv::Scalar(0, 0, 255), 2);
    }




    video_writer_->write(frame1);

    // std::vector<edk::DetectObject> arrDetection;
    // std::vector<std::vector<DetectedObject>> arrDetection;
    // std::string img_name1 = "/data/wcl/174/wcl/data/gongdi/image/1_27144.jpg";
    // std::string img_name2 = "/data/wcl/174/wcl/data/gongdi/image/1_34608.jpg";
    // std::string img_name3 = "/data/wcl/174/wcl/data/gongdi/image/2_2736.jpg";
    // std::string img_name4 = "/data/wcl/174/wcl/data/gongdi/image/2_2712.jpg";
    // cv::Mat frame1 = cv::imread(img_name1);
    // cv::Mat frame2 = cv::imread(img_name2);
    // cv::Mat frame3 = cv::imread(img_name3);
    // cv::Mat frame4 = cv::imread(img_name4);
    // struct timeval time1, time2, time3, time4;

    // std::vector<cv::Mat> batch_image;
    // // for(int jj = 0;jj<4;jj++)
    // // {
    // batch_image.push_back(frame1);
    // batch_image.push_back(frame2);
    // batch_image.push_back(frame3);
    // batch_image.push_back(frame4);

    // gettimeofday(&time1, NULL);
    // std::vector<std::vector<cv::Mat>> preprocessedImages;
    // std::vector<cv::Size> image_paras;
    // g_runner->Pre(batch_image, preprocessedImages, image_paras);
    // gettimeofday(&time2, NULL);
    // double pre_time = ((time2.tv_sec - time1.tv_sec) * 1000000.0f + time2.tv_usec - time1.tv_usec) / 1000.0f;
    // LOG(INFO) << " pre time: " << pre_time << "ms";
    // g_runner->Detect(preprocessedImages, arrDetection, image_paras);
    // gettimeofday(&time3, NULL);
    // double one_time = ((time3.tv_sec - time2.tv_sec) * 1000000.0f + time3.tv_usec - time2.tv_usec) / 1000.0f;
    // LOG(INFO) << " one time: " << one_time << "ms";

    // gettimeofday(&time4, NULL);
    // double aver_time = ((time4.tv_sec - time1.tv_sec) * 1000000.0f + time4.tv_usec - time1.tv_usec) / 1000.0f;
    // LOG(INFO) << " aver time: " << aver_time / 1000 << "ms";

    // cv::Mat image_copy = frame.clone();
    // for (int j = 0; j < 4; j++) {
    //   int len = arrDetection[j].size();
    //   std::cout << "result len:" << len << std::endl;
    //   for (int i = 0; i < len; i++) {
    //     float x0 = arrDetection[j][i].bounding_box.x;
    //     float y0 = arrDetection[j][i].bounding_box.y;
    //     float x1 = (arrDetection[j][i].bounding_box.x + arrDetection[j][i].bounding_box.width);
    //     float y1 = (arrDetection[j][i].bounding_box.y + arrDetection[j][i].bounding_box.height);
    //     int label = arrDetection[j][i].object_class;
    //     float pro = arrDetection[j][i].prob;
    //     cv::Point p1(x0, y0);
    //     cv::Point p2(x1, y1);
    //     cv::Point p5(x0, y0 + 10);
    //     cv::Point p6(x0, y0 + 30);
    //     // cv::resize(image_copy, image_copy, cv::Size(416, 416));
    //     cv::rectangle(batch_image[j], p1, p2, cv::Scalar(0, 0, 255), 2);
    //     cv::putText(batch_image[j], std::to_string(label), p5, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(240, 240, 0),
    //                 1);
    //     cv::putText(batch_image[j], std::to_string(pro), p6, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(240, 240, 0), 1);
    //     cv::imwrite("/data/wcl/code/easydk_61/samples/test/" + std::to_string(j) + ".jpg", batch_image[j]);
    //     // exit(0);
    //   }
    // }
    
    // cv::imwrite(savename, image_copy);
    // video_writer_->write(image_copy);
  }
  
  video_writer_->release();
  
  return 0;
}
