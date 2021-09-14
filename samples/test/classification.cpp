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
#include <unistd.h>

#include <csignal>
#include <future>
#include <iostream>
#include <memory>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>
#include <string>
#include <vector>
#include "device/mlu_context.h"
#include "inference.h"

#include "cnpostproc.h"
#include "device/mlu_context.h"
#include "easybang/resize_and_colorcvt.h"
#include "easycodec/easy_decode.h"
#include "easyinfer/easy_infer.h"
#include "easyinfer/mlu_memory_op.h"
#include "easyinfer/model_loader.h"
#include "easytrack/easy_track.h"
#include <sys/time.h>
DEFINE_int32(repeat_time, 0, "process repeat time");
DEFINE_string(data_path, "", "video path");
DEFINE_string(model_path, "", "infer offline model path");

DEFINE_string(func_name, "subnet0", "model function name");
DEFINE_int32(wait_time, 0, "time of one test case");

int main(int argc, char **argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // check params
  CHECK_NE(FLAGS_data_path.size(), 0u);
  CHECK_NE(FLAGS_model_path.size(), 0u);
  CHECK_NE(FLAGS_func_name.size(), 0u);

  CHECK_GE(FLAGS_wait_time, 0);
  CHECK_GE(FLAGS_repeat_time, 0);
  std::cout << "FLAGS_func_name:" << FLAGS_func_name << std::endl;
  std::string names_file = "/data/zhn/libc/names.txt";
  std::shared_ptr<ClassificationRunner> g_runner = std::make_shared<ClassificationRunner>(FLAGS_model_path, FLAGS_func_name, names_file, 0);

  std::cout << "initial success" << std::endl;

  std::string img_name1 = "/data/zhn/libc/img/green/93_green.jpg";
  cv::Mat frame1 = cv::imread(img_name1);
  std::vector<cv::Mat> batch_image;
  batch_image.push_back(frame1);

  std::vector<std::vector<Prediction>> result;
  result = g_runner->Classify(batch_image, 1);

  for (int g = 0; g < 1; ++g)
  {
    std::cout << "label:" << result[g][0].first << std::endl;
    std::cout << "confidence:" << result[g][0].second << std::endl;
  }

  // std::string imagepath = FLAGS_data_path;
  // // string imagepath = "/data/models/wcl/data/yiwu/3-chibo/image.txt";
  // std::string const folder = imagepath.substr(0, imagepath.find_last_of("/"));
  // FILE *pFile = fopen(imagepath.c_str(), "r");

  // if (NULL == pFile)
  //   return -1;
  // char path[256] = {0};
  // // set mlu environment
  // while (fscanf(pFile, "%[^\n]%*c", path) != EOF)
  // {
  //   std::string strpath = folder + "/" + path;
  //   //videoname_without_ext   .../folder/video
  //   std::string const videoname_without_ext = strpath.substr(0, strpath.find_last_of("."));
  //   cv::Mat frame = cv::imread(strpath);
  //   // std::vector<edk::DetectObject> arrDetection;
  //   std::vector<std::vector<DetectedObject>> arrDetection;
  //   // g_runner->Process(frame,arrDetection);

  //   std::string img_name1 = "/data/zhn/libc/img/green/93_green.jpg";
  //   std::string img_name2 = "/data/zhn/libc/img/green/103_110010.jpg";
  //   std::string img_name3 = "/data/zhn/libc/img/green/103_150010.jpg";
  //   std::string img_name4 = "/data/zhn/libc/img/green/103_170010.jpg";
  //   cv::Mat frame1 = cv::imread(img_name1);
  //   cv::Mat frame2 = cv::imread(img_name2);
  //   cv::Mat frame3 = cv::imread(img_name3);
  //   cv::Mat frame4 = cv::imread(img_name4);
  //   std::vector<cv::Mat> batch_image;
  //   batch_image.push_back(frame1);
  //   batch_image.push_back(frame2);
  //   batch_image.push_back(frame3);
  //   batch_image.push_back(frame4);

  //   int i = 0;
  //   struct timeval time1, time2, time3, time4;
  //   gettimeofday(&time1, NULL);
  //   std::vector<std::vector<Prediction>> result;
  //   while (1)
  //   {
  //     gettimeofday(&time2, NULL);
  //     result = g_runner->Classify(batch_image);
  //     gettimeofday(&time3, NULL);
  //     double one_time = ((time3.tv_sec - time2.tv_sec) * 1000000.0f + time3.tv_usec - time2.tv_usec) / 1000.0f;
  //     LOG(INFO) << " one time: " << one_time << "ms";
  //     i++;
  //     // if(i==1000)
  //     //   break;
  //   }
  //   gettimeofday(&time4, NULL);
  //   double aver_time = ((time4.tv_sec - time1.tv_sec) * 1000000.0f + time4.tv_usec - time1.tv_usec) / 1000.0f;
  //   LOG(INFO) << " aver time: " << aver_time / 1000 << "ms";
  //   // return 0;

  //   for (int g = 0; g < 4; ++g)
  //   {
  //     std::cout << "label:" << result[g][0].first << std::endl;
  //     std::cout << "confidence:" << result[g][0].second << std::endl;
  //   }

  // std::cout << "len: " << arrDetection.size() << std::endl;
  //     for (int j=0;j<4;++j){
  //       std::cout << "len: " << arrDetection[j].size() << std::endl;
  //         for (auto& obj : arrDetection[j]) {

  //         std::cout << "[Object] label: " << obj.object_class << " score: " << obj.prob << "\n";
  //       }
  //     }
  //   return 0;
  // }
  return 0;
}
