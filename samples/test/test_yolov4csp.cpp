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

using namespace std;
using namespace cv;



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
    }
    else {
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
#define detection
int main()
{
    #ifdef detection
                                                                        //data1/zhn/suguan/yolov3_4b4c_simple.cambricon /data/ld/project/yolov4-csp-mlu/yolov4_int8_4_4.cambricon
    std::shared_ptr<Detection> g_runner = std::make_shared<Detection>("/data1/zhn/suguan/yolov3_4b4c_simple.cambricon", "subnet0", 0,0);
    #else
    std::shared_ptr<DetectionRunner> g_runner = std::make_shared<DetectionRunner>("/data/ld/project/darknet_scaled_yolov4/yolov4csp.cambricon", "subnet0", 0);
    #endif
    cv::Size sizeGrunner = cv::Size(416,416);
    std::string videoTxt = "/data/ld/test_video/20220707/video.txt";
    std::fstream file;
    file.open(videoTxt,std::ios_base::in);

    std::string videoName;
    while (getline(file, videoName))
    {
        std::string savename = videoName.substr(0,videoName.find(".mp4",0))+".avi";
        cv::VideoCapture cap(videoName);
        Size videoSize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
        cv::VideoWriter writer;
        writer.open(savename, VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, videoSize);
        cv::Mat img1_resize;
        //cv::Mat img1 = cv::imread("/data/ld/project/mlu-pytorch-alphapose-master/1.jpg");
        cv::Mat img1;
        int currentFrame = 0;
        int skip = 1;
        while (true)
        {
            currentFrame += 1;
            if (currentFrame % skip != 0)
            {
                continue;
            }
            if (!cap.read(img1))
                break;
            std::vector<cv::Mat> batch_image;
            #ifdef detection
            preYolov3(img1, sizeGrunner.width, sizeGrunner.height, img1_resize);
            //cv::resize(img1,img1_resize,cv::Size(416,416));
            #else
            std::vector<std::vector<cv::Mat>> preprocessedImages;
            cv::resize(img1,img1,cv::Size(416,416));
            std::vector<cv::Size> image_paras;
            #endif
            std::vector<cv::Size> vecSize;
            std::vector<std::vector<DetectedObject>> arrDetection;
            #ifdef detection
            for(int i = 0; i < 4; i++)
            {
                batch_image.push_back(img1_resize);
                vecSize.push_back(cv::Size(img1.cols, img1.rows));
            }
            #else
            batch_image.push_back(img1);
            #endif
            double time0 = static_cast<double>(cv::getTickCount());
            #ifdef detection
            g_runner->Detect(batch_image, arrDetection, vecSize);            
            #else
            g_runner->Pre(batch_image, preprocessedImages, image_paras);
            g_runner->Detect(preprocessedImages, arrDetection, image_paras);
            #endif
            time0 = ((double)cv::getTickCount() - time0) / cv::getTickFrequency();
            std::cout << "此方法运行时间 为:" << time0 *1000 << "毫秒" << std::endl;
            std::vector<cv::Mat> imgs;
            cv::Mat imgclone = img1.clone();
            std::vector<std::vector<cv::Point>> pts;
            std::cout<<"arrDetection[0].size(): "<<arrDetection[0].size()<<std::endl;
            for (int i = 0; i < arrDetection[0].size(); i++)
            {
                //std::cout<<"arrDetection object_class: "<<arrDetection[0][i].object_class<<std::endl;
                cv::Rect rect = arrDetection[0][i].bounding_box;
                cv::rectangle(imgclone, rect, cv::Scalar(0, 128, 255), 2);
                std::string txt = std::to_string(arrDetection[0][i].object_class) + "_" + std::to_string(arrDetection[0][i].prob);
                cv::putText(imgclone, txt, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX,1, cv::Scalar(0, 0, 255), 3, 8);
            }
            writer.write(imgclone);
            }
    }
    // cv::rectangle(img1, rect, cv::Scalar(0, 128, 255), 2);
    // std::string txt = std::to_string(arrDetection[0][nn].object_class) + "_" + std::to_string(arrDetection[0][nn].prob);
    // cv::putText(img1, txt, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 3, 8);

// for(int i = 0; i < alphaPose.GetBatch(); i++)
// {
//     imgs.push_back(img);
// }

// for(int i = 0; i < 1; i++)
// {
//     cv::Mat imgclone = img.clone();
//     double time0 = static_cast<double>(cv::getTickCount());
//     std::vector<DetectedObject> arrDetection;

//     alphaPose.getKeyPoints(img,arrDetection,pts);
//     time0 = ((double)cv::getTickCount() - time0) / cv::getTickFrequency();
//     std::cout << "此方法运行时间为:" << time0 *1000 << "毫秒" << std::endl;
//     for(int i = 0; i < pts.size(); i++)
//     {
//         std::vector<cv::Point> batchImgPts = pts[i];
//         for(auto& pt : batchImgPts)
//         {
//             cv::circle(imgclone,pt,5, cvScalar(0, 0, 255), 2);
//         }
//         cv::imwrite("/data/ld/project/mlu-pytorch-alphapose-master/res_"+std::to_string(i)+".jpg",imgclone);
//     }
//     pts.clear();
// }
    return 0;
}