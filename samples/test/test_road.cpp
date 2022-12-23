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


double test(cv::Mat img1,cv::Mat img2)
{
     return cv::matchShapes(img1,img2,CONTOURS_MATCH_I2,0);
}


void isRoad(const cv::Mat& mask, bool& flag,  cv::Mat& processedMask,std::vector<std::vector<cv::Point>>& printContours,float arearate = 3.0 , float ratethresold = 1.0)
{
    flag = false;
    //cv::Mat thresholdImg;
    cv::Mat resMask(mask.size(),CV_8UC1);
    resMask.setTo(0);
    //cv::threshold(mask,thresholdImg,120,255,cv::THRESH_BINARY);
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> remainedcontours;
    std::vector<std::vector<cv::Point>> approxs;
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    printContours = contours;
    // for(auto& contour : contours)
    // {
    //     float area = cv::contourArea(contour);
    //     if (area * 100 / (mask.cols * mask.rows) > arearate)
    //     {
    //         remainedcontours.push_back(contour);
    //         double epsilon = 1;
    //         std::vector<cv::Point> pts;
    //         cv::approxPolyDP(contour,pts,epsilon,true);
    //         approxs.push_back(pts);
    //     }
    // }
    // for(auto& approx : approxs)
    // {
    //     cv::RotatedRect  rotated_box = cv::minAreaRect(approx);
    //     if(rotated_box.size.height * rotated_box.size.width != 0 )
    //     {
    //         float rate = rotated_box.size.height / rotated_box.size.width;
    //         if(rate < 1 )
    //         {
    //             rate = 1 / rate;
    //         }
    //         if(rate > ratethresold)
    //         {
    //             flag = true;
    //         }
    //     }
    // }
    for(auto& contour : contours)
    {
        float area = cv::contourArea(contour);
        if (area * 100 / (mask.cols * mask.rows) > arearate)
        {
            remainedcontours.push_back(contour);
            // double epsilon = 1;
            // std::vector<cv::Point> pts;
            // cv::approxPolyDP(contour,pts,epsilon,true);
            // approxs.push_back(pts);
        }
    }
    for(auto& approx : remainedcontours)
    {
        cv::RotatedRect  rotated_box = cv::minAreaRect(approx);
        if(rotated_box.size.height * rotated_box.size.width != 0 )
        {
            float rate = rotated_box.size.height / rotated_box.size.width;
            if(rate < 1 )
            {
                rate = 1 / rate;
            }
            if(rate > ratethresold)
            {
                flag = true;
            }
        }
    }
    cv::drawContours(resMask,remainedcontours,-1,cv::Scalar(255),-1);
    processedMask = resMask;
}


void isRoad(const cv::Mat& mask, bool& flag,  cv::Mat& processedMask,float arearate = 7.0 ,float ratethresold = 2)
{
    flag = false;
    //cv::Mat thresholdImg;
    cv::Mat resMask(mask.size(),CV_8UC1);
    resMask.setTo(0);
    //cv::threshold(mask,thresholdImg,120,255,cv::THRESH_BINARY);
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> remainedcontours;
    std::vector<std::vector<cv::Point>> approxs;
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    for(auto& contour : contours)
    {
        float area = cv::contourArea(contour);
        if (area * 100 / (mask.cols * mask.rows) > arearate)
        {
            remainedcontours.push_back(contour);
            // double epsilon = 1;
            // std::vector<cv::Point> pts;
            // cv::approxPolyDP(contour,pts,epsilon,true);
            // approxs.push_back(pts);
        }
    }
    for(auto& approx : remainedcontours)
    {
        cv::RotatedRect  rotated_box = cv::minAreaRect(approx);
        if(rotated_box.size.height * rotated_box.size.width != 0 )
        {
            float rate = rotated_box.size.height / rotated_box.size.width;
            if(rate < 1 )
            {
                rate = 1 / rate;
            }
            if(rate > ratethresold)
            {
                flag = true;
            }
        }
    }
    cv::drawContours(resMask,remainedcontours,-1,cv::Scalar(255),-1);
    processedMask = resMask;
}



void camerarotation(const std::vector<cv::Mat>& masks,bool& flag , cv::Mat& processedMask, float arearate = 3.0 , float ratethresold = 1.0)
{
    flag = false;
    int falseNum = 0;
    const float iou = 90;
    const float isNotRoadRate = 10;
    std::vector<cv::Mat> road_masks;
    cv::Mat final_road_mask;
    for (auto& mask : masks)
    {
        isRoad(mask,flag,processedMask,arearate,ratethresold);

        if (!flag)
            falseNum+=1;
        else
            road_masks.push_back(processedMask);
    }
    if(falseNum * 100 / masks.size() > isNotRoadRate)
    {
        flag = true;
        //processedMask = final_road_mask;
    }
    else
    {
        for(int i = 0; i < road_masks.size(); i++)
        {
            cv::imwrite("/data/ld/test_video/20220629/road/逆行自动识别视频-倒放/road_mask_"+std::to_string(i)+".jpg",road_masks[i]);
        }
        cv::Mat pre;
        int intersectNum = 0;
        //double sim ;
        for(int i = 0; i < road_masks.size();i++)
        {
            if(i!=0)
            {     
                float mask_and_area = 0;
                float mask_or_area = 0;
                // cv::resize(road_masks[i],road_masks[i],cv::Size(128,128));
                // cv::resize(pre,pre,cv::Size(128,128));
                cv::Mat mask_and,mask_or;
                cv::bitwise_and(road_masks[i],pre,mask_and);
                cv::bitwise_or(road_masks[i],pre,mask_or);
                std::vector<std::vector<cv::Point>> mask_and_contours;
                std::vector<std::vector<cv::Point>> mask_or_contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(mask_and,mask_and_contours,hierarchy,cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
                cv::findContours(mask_or,mask_or_contours,hierarchy,cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
                for (int j = 0; j < mask_and_contours.size();j++)
                {
                    mask_and_area+=cv::contourArea(mask_and_contours[j]);

                }
                for (int j = 0; j < mask_or_contours.size();j++)
                {
                    mask_or_area+=cv::contourArea(mask_or_contours[j]);

                }
                if(mask_and_area*mask_or_area!=0 and mask_and_area*100/mask_or_area>iou)
                {
                    cv::imwrite("/data/ld/test_video/20220629/road/逆行自动识别视频-倒放/mask_and_"+std::to_string(i)+".jpg",mask_and);
                    cv::imwrite("/data/ld/test_video/20220629/road/逆行自动识别视频-倒放/mask_or_"+std::to_string(i)+".jpg",mask_or);
                    //std::cout<<"current iou rate: "<<mask_and_area*100/mask_or_area<<std::endl;
                    intersectNum+=1;
                }
            }
            if(i ==0)
            {
                pre = road_masks[i];
            }
        }
        //std::cout<<"相交个数比例: "<<intersectNum * 100 / (road_masks.size() - 1)<<std::endl;
        //if(intersectNum * 100 / (road_masks.size() - 1) > 90)
        if(intersectNum == road_masks.size()-1)
        {
            for (int i = 0; i < road_masks.size(); i++ )
            {
                final_road_mask = road_masks[i];
                if(i != 0)
                {
                    cv::bitwise_or(pre,road_masks[i],final_road_mask);
                }
                pre = final_road_mask;
            }
            flag = false;
            processedMask = final_road_mask;
            //processedMask = road_masks[road_masks.size()-1];
        }
        else
        {
            flag = true;
        }
        
    }
    
}


void get_road_coordinate(cv::Mat& final_road_mask,cv::Mat& frame,std::vector<std::vector<cv::Point>>& approxs)
{
    approxs.clear();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(final_road_mask,contours,hierarchy,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);
    for(auto& contour : contours)
    {
        double epsilon = 1;
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour,approx,epsilon,true); 
        for(auto& pt : approx )
        {
            pt.x = pt.x * frame.cols / final_road_mask.cols;
            pt.y = pt.y * frame.rows / final_road_mask.rows;
        }
        approxs.push_back(approx);
    }
}

void road_test()
{
    std::string modelPath = "/data/ld/project/instanceseg_new/DDRmodel/v5/road.cambricon";
    std::string funcname = "subnet0";
    int gpuid = 0;
    fpnSegment seg(modelPath,funcname,gpuid);
    std::string videotxt = "/data/ld/test_video/20220704/video/拥堵/video.txt";
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
        int currentFrame = 0;
        int skip = 27;
        int cacheNum = 5;
        std::vector<cv::Mat> caches; 
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
            if (currentFrame%skip!=0)
                continue;
            double time0 = static_cast<double>(cv::getTickCount());
            seg.getfeat(frame,feat); 
            
            time0 = ((double)cv::getTickCount() - time0) / cv::getTickFrequency();
            std::cout << "此方法运行时间为:" << time0 *1000 << "毫秒" << std::endl;
            caches.push_back(feat.clone());
            frames.push_back(frame.clone());
            //cv::imwrite("/data/ld/test_video/1.jpg",frame);   
            if(caches.size() == cacheNum)
            {
                bool flag = false;
                camerarotation(caches,  flag , processedMask);
                // for(int i = 0; i < caches.size(); i++)
                // {
                //     cv::imwrite("/data/ld/test_video/"+std::to_string(i)+".png",caches[i]);

                // }
                int index = 0;
                bool isroad = false;
                std::vector<std::vector<cv::Point>> approxs;
                for(int i = 0; i < frames.size(); i++)
                {
                    // if(currentFrame - cacheNum + i == 1888)
                    // {
                    //     // cv::imwrite("/data/ld/test_video/test.png",frames[i]);
                    //     // return 0;
                    //     isRoad(caches[i],isroad,processedMask);
                    // }
                    cv::Mat tmp;
                    std::vector<std::vector<cv::Point>> printContours;
                    std::vector<float> areaRates;
                    std::vector<float> rotateRates;
                    isRoad(caches[i],isroad,tmp,printContours);
                    for(auto& contour : printContours)
                    {
                        float area = cv::contourArea(contour);
                        float areaRate = area * 100 / (seg.out_h * seg.out_w) ;
                        areaRates.push_back(areaRate);
                        cv::RotatedRect rotated_box = cv::minAreaRect(contour);
                        float rate = rotated_box.size.height / rotated_box.size.width;
                        if(rate < 1 )
                        {
                            rate = 1 / rate;
                        }
                        rotateRates.push_back(rate);
                        for(auto& c : contour)
                        {
                            c.x = frames[i].cols * c.x / seg.out_w;
                            c.y = frames[i].rows * c.y / seg.out_h;
                        }
                    }
                    
                    //get_road_coordinate(caches[i],frames[i],approxs);
                    //drawContours(frames[i], approxs, -1, Scalar(0, 0, 255), 2);
                    //approxs.clear();
                
                    // for(auto& approx : approxs)
                    // {
                    //     std::cout<<"road point size is : "<<approx.size()<<std::endl;
                    // }
                    
                 
                    if(flag)
                    {
                        drawContours(frames[i], printContours, -1, Scalar(0, 255, 0), 2);
                        for(int j = 0; j < printContours.size(); j++)
                        {
                            std::vector<cv::Point> pts = printContours[j];
                            cv::RotatedRect rotated_box = cv::minAreaRect(pts);
                            // putText(frames[i],"areaRates: "+std::to_string(areaRates[i]),rotated_box.center, FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 0, 255), 2, 1);
                            // putText(frames[i],"rotateRates: "+std::to_string(rotateRates[i]),cv::Point(rotated_box.center.x - 5,rotated_box.center.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2, 1);
                            putText(frames[i],"areaRates: "+std::to_string(areaRates[j]),rotated_box.center, FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 0, 255), 2, 1);
                            putText(frames[i],"rotateRates: "+std::to_string(rotateRates[j]),cv::Point(rotated_box.center.x - 20,rotated_box.center.y - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2, 1);
                        }
                        // get_road_coordinate(processedMask,frames[i],approxs);
                        // approxs.clear();
                        putText(frames[i], "frame: " + std::to_string(index++) +" "+"zhuan dong: " +std::to_string(printContours.size()) , Point((frame.cols / 2), frame.rows / 2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2, 1);
                    }
                    else
                    {
                        get_road_coordinate(processedMask,frames[i],approxs);
                        drawContours(frames[i], approxs, -1, Scalar(0, 0, 255), 2);
                        approxs.clear();
                        putText(frames[i], "frame: " + std::to_string(index++) +" " + "jing zhi", Point((frame.cols / 2), frame.rows / 2), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2, 1);
                    }
                    if(isroad)
                    {
                        putText(frames[i], "is road" , Point((frame.cols / 2) , frame.rows / 2 - 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2, 1);
                    }
                    else
                    {
                        putText(frames[i], "is not road" , Point((frame.cols / 2) , frame.rows / 2 - 100), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 2, 1);
                    }
                    putText(frames[i], "current Frame id: " + std::to_string(currentFrame - cacheNum + i), Point((frame.cols / 2) , frame.rows / 2 - 150), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2, 1);
                    writer.write(frames[i]);
                }
                caches.clear();
                frames.clear();
            }
        }
    }

}

int main()
{   
   road_test();
}