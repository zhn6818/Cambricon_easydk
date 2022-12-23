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


int main(int argc, char **argv) {
    //const std::string prefix = "/data/ld/project/mtcnn/cnmodels";
       // std::vector<std::string> pnet_model_path{prefix + "/det1_384x216_int8.cambricon",
    //                                          prefix + "/det1_288x162_int8.cambricon",
    //                                          prefix + "/det1_216x122_int8.cambricon",
    //                                          prefix + "/det1_162x92_int8.cambricon",
    //                                          prefix + "/det1_122x69_int8.cambricon",
    //                                          prefix + "/det1_92x52_int8.cambricon",
    //                                          prefix + "/det1_69x39_int8.cambricon",
    //                                          prefix + "/det1_52x29_int8.cambricon",
    //                                          prefix + "/det1_39x22_int8.cambricon",
    //                                          prefix + "/det1_29x17_int8.cambricon"};
    const std::string modelfolder = "/data/ld/project/mtcnn/mtcnn_models_new_version";
    // std::vector<std::string> pnet_model_path{prefix + "/det1_576x324_int8.cambricon",
    //                                          prefix + "/det1_409x230_int8.cambricon",
    //                                          prefix + "/det1_290x163_int8.cambricon",
    //                                          prefix + "/det1_206x116_int8.cambricon",
    //                                          prefix + "/det1_146x82_int8.cambricon",
    //                                          prefix + "/det1_104x59_int8.cambricon",
    //                                          prefix + "/det1_74x42_int8.cambricon",
    //                                          prefix + "/det1_52x30_int8.cambricon",
    //                                          prefix + "/det1_37x21_int8.cambricon",
    //                                          prefix + "/det1_27x15_int8.cambricon"};

    // std::string rnet_model_path = prefix + "/det2_16batch_int8.cambricon";
    // std::string onet_model_path = prefix + "/det3_16batch_int8.cambricon";
    OpencvMtcnn mtcnn;
    mtcnn.load_model(modelfolder, "subnet0", 0);
    //cv::Mat mat = cv::imread("/data/ld/project/mtcnn/33.jpg");
    //cv::Mat mat = cv::imread("/data/ld/project/mtcnn/face.png");
    //cv::resize(mat,mat,cv::Size(1920,1080));
    std::string videopath = "/data/ld/project/mtcnn/2.mp4";
    std::string videowritepath = "/data/ld/project/mtcnn/res.avi";
    cv::VideoWriter videowriter;
    cv::VideoCapture capture(videopath);
    if(!capture.isOpened())
    {
      std::cout<<"视频不能打开"<<std::endl;
    }
    cv::Size videosize(capture.get(3),capture.get(4));
    videowriter.open(videowritepath,cv::VideoWriter::fourcc('D','I','V','X'),20,videosize);
    cv::Mat mat;
    while(true)
    {
      if(!capture.read(mat))
          break;
      std::vector<face_box> face_list;
      mtcnn.detect(mat, face_list);
      for(int i = 0; i < face_list.size(); i++)
      {
        face_box face = face_list[i];
        cv::Rect rect = cv::Rect(face.x0, face.y0, face.x1 - face.x0, face.y1 - face.y0);
        cv::rectangle(mat, rect, cv::Scalar(0, 0, 255));
        for (int i = 0;i < 5;i++)
        {
            cv::circle(mat, cv::Point(face.landmark.x[i], face.landmark.y[i]), 5, cv::Scalar(0, 255, 0), 2);
        }
      }
      videowriter.write(mat);
    }

    //cv::imwrite("/data/ld/project/mtcnn/res.jpg", mat);
    
    return 0;
}
