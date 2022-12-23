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

using namespace cv;
int main()
{
    std::string modelPath = "/data/ld/project/instanceseg_new/DDRmodel/v7/fog.cambricon";
    std::string funcname = "subnet0";
    int gpuid = 0;
    fpnSegment seg(modelPath,funcname,gpuid);
    std::string videoTxt = "/data/ld/test_video/video.txt";
    std::fstream file;
    file.open(videoTxt,std::ios_base::in);
    std::string videoName;
    int currentFrame = 0;
    int skipFrame = 25;
    while (getline(file, videoName))
    {
        std::string savename = videoName.substr(0,videoName.find(".mp4",0))+".avi";
        cv::VideoCapture cap(videoName);
        Size videoSize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
        cv::VideoWriter writer;
        writer.open(savename, VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, videoSize);
        cv::Mat frame;
        currentFrame = 0;
        while (true)
        {
            if (!cap.read(frame))
                break;
            currentFrame+=1;
            if(!(currentFrame % skipFrame == 0))
            {
                continue;
            }
            std::vector<cv::Mat> matVec;
            matVec.push_back(frame);
            cv::Mat feat;
            double time0 = static_cast<double>(cv::getTickCount());
            seg.getfeat(frame, feat);
            time0 = ((double)cv::getTickCount() - time0) / cv::getTickFrequency();
            std::cout << "此方法运行时间 为:" << time0 *1000 << "毫秒" << std::endl;
            bool isSmoke;
            int binary_pixels;
            cv::Mat dstImg;
            std::vector<cv::Point>pts;
            pts = { Point(0,0),Point(frame.cols - 1,0),Point(frame.cols - 1,frame.rows - 1),Point(0,frame.rows - 1),Point(0,0) };
            std::vector<std::vector<cv::Point>> contours;
            std::vector<std::vector<cv::Point>> contours_all;
            seg.processFeat_test(feat, pts, videoSize, 0, contours, isSmoke, binary_pixels, contours_all, dstImg);
            for (int i = 0;i < contours.size();i++)
            {
                polylines(frame, contours[i], true, cv::Scalar(0, 0, 255), 2);
            }
            for (int i = 0;i < contours_all.size();i++)
            {
                polylines(frame, contours_all[i], true, cv::Scalar(0, 255, 0), 2);
            }
            char showMsg[256] = { 0 };
            sprintf(showMsg, "isSmoke:%d , %d", isSmoke, binary_pixels);
            putText(frame, showMsg, Point(100, 100 + 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2, 1);
            writer.write(frame);
        }

    }

 
    
}