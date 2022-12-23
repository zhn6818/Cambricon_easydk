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
#include <fstream>
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

// void isRoadv1(const cv::Mat& mask, bool& flag,  cv::Mat& processedMask, int areathresold = 5000,float ratethresold = 3)
// {
//     flag = false;
//     //cv::Mat thresholdImg;
//     cv::Mat resMask(mask.size(),CV_8UC1);
//     resMask.setTo(0);
//     //cv::threshold(mask,thresholdImg,120,255,cv::THRESH_BINARY);
//     std::vector<cv::Vec4i> hierarchy;
//     std::vector<std::vector<cv::Point>> contours;
//     std::vector<std::vector<cv::Point>> remainedcontours;
//     std::vector<std::vector<cv::Point>> approxs;
//     cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
//     for(auto& contour : contours)
//     {
//         for (auto& point: contour)
//         {
//             point.x = point.x * 256 / mask.cols;
//             point.y = point.y * 256 / mask.rows;
//         }
//         int area = cv::contourArea(contour);
//         if (area > areathresold)
//         {
//             remainedcontours.push_back(contour);
//             double epsilon = 1;
//             std::vector<cv::Point> pts;
//             cv::approxPolyDP(contour,pts,epsilon,true);
//             approxs.push_back(pts);
//         }
//     }
//     for(auto& approx : approxs)
//     {
//         cv::RotatedRect  rotated_box = cv::minAreaRect(approx);
//         if(rotated_box.size.height * rotated_box.size.width != 0 )
//         {
//             float rate = rotated_box.size.height / rotated_box.size.width;
//             if(rate < 1 )
//             {
//                 rate = 1 / rate;
//             }
//             if(rate > ratethresold)
//             {
//                 flag = true;
//             }
//         }
//     }
//     if(flag)
//     {
//         for(auto& approx : approxs)
//         {
//             for(auto& pt : approx )
//             {
//                 pt.x = pt.x * mask.cols / 256 ;
//                 pt.y = pt.y * mask.rows / 256 ;
//             }
//         }
//     }
//     cv::drawContours(resMask,approxs,-1,cv::Scalar(255),-1);
//     processedMask = resMask;
// }

double test(cv::Mat img1,cv::Mat img2)
{
     return cv::matchShapes(img1,img2,CONTOURS_MATCH_I2,0);
}



void road_test()
{
    std::string modelPath = "/data/ld/project/instanceseg_new/DDRmodel/v7/road.cambricon";
    std::string funcname = "subnet0";
    int gpuid = 0;
    fpnSegment seg(modelPath,funcname,gpuid);
    std::string videotxt = "/data/ld/test_video/video.txt";
    std::ifstream  ifs;
    ifs.open(videotxt,std::ios::in);
    if (!ifs.is_open())
	{
		std::cout << "打开文件失败！！！";
		return ;
	}
    std::string temp;
    std::string videoName ;
    while (getline(ifs, videoName))
	{
        std::string savename = videoName.substr(0,videoName.find(".mp4",0))+".avi";
        cv::VideoCapture cap(videoName);
        Size videoSize(cap.get(CAP_PROP_FRAME_WIDTH),cap.get(CAP_PROP_FRAME_HEIGHT));
        cv::VideoWriter writer;
        writer.open(savename,VideoWriter::fourcc('D','I','V','X'),20, videoSize);
        cv::Mat frame;
        int currentFrame = 1;
        int skip = 1;
        int cacheNum = 5;
        std::vector<cv::Mat> frames;
        cv::Mat processedMask;
        while (true)
        {
            if(!cap.read(frame))
                break;
            std::vector<cv::Mat> matVec;
            //frame = cv::imread("/data/ld/test_video/test.png");
            matVec.push_back(frame);
            cv::Mat feat;
            currentFrame++;
            if (!(currentFrame%skip==0))
                continue;
            double time0 = static_cast<double>(cv::getTickCount());
            seg.getfeat(frame,feat);
            cv::imwrite("/data/ld/test_video/20220629/road/逆行自动识别视频-倒放/res.jpg",feat);
            time0 = ((double)cv::getTickCount() - time0) / cv::getTickFrequency();
            std::cout << "此方法运行时间为:" << time0 *1000 << "毫秒" << std::endl;
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(feat,contours,hierarchy,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);
            for(int i = 0; i < contours.size(); i++)
            {
                for (int j = 0; j < contours[i].size(); j++)
                {
                    contours[i][j].x *= ((float)frame.cols / feat.cols);
                    contours[i][j].y *= ((float)frame.rows / feat.rows);
                }
            }
            drawContours(frame, contours, -1, Scalar(0, 0, 255), 2);
            //cv::imwrite("/data/ld/test_video/20220707/res.jpg",frame);
            writer.write(frame);
            }
        }
    }


int main()
{   
   road_test();
}