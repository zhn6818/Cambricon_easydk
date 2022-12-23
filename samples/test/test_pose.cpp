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

int main()
{

    std::shared_ptr<Detection> g_runner = std::make_shared<Detection>("/data1/zhn/suguan/yolov3_4b4c_simple.cambricon", "subnet0", 0);
    cv::Size sizeGrunner = g_runner->GetSize();
    std::string videoName = "/data/ld/project/mlu-pytorch-alphapose-master/3_safty-helmet2021-10-20-09-43-01.mp4";
    std::string savename = "/data/ld/project/mlu-pytorch-alphapose-master/3_safty-helmet2021-10-20-09-43-01.avi";
    cv::VideoCapture cap(videoName);
    Size videoSize(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer;
    writer.open(savename, VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, videoSize);
    cv::Mat img1_resize;
    //cv::Mat img1 = cv::imread("/data/ld/project/mlu-pytorch-alphapose-master/1.jpg");
    cv::Mat img1;
    int currentFrame = 0;
    string modelPath = "/data/ld/project/mlu-pytorch-alphapose-master/mluModel/pose.cambricon";
    string func_name = "subnet0";
    int device_id = 0;
    AlphaPose alphaPose = AlphaPose(modelPath, func_name, device_id);
    while (true)
    {
        currentFrame += 1;
        if (currentFrame % 20 != 0)
        {
            continue;
        }
        if (!cap.read(img1))
            break;
        preYolov3(img1, sizeGrunner.width, sizeGrunner.height, img1_resize);
        std::vector<cv::Size> vecSize;
        vecSize.push_back(cv::Size(img1.cols, img1.rows));
        std::vector<std::vector<DetectedObject>> arrDetection;
        std::vector<cv::Mat> batch_image;
        batch_image.push_back(img1_resize);
        batch_image.push_back(img1_resize);
        batch_image.push_back(img1_resize);
        batch_image.push_back(img1_resize);
        g_runner->Detect(batch_image, arrDetection, vecSize);
        std::vector<cv::Mat> imgs;
        cv::Mat imgclone = img1.clone();
        std::vector<std::vector<cv::Point>> pts;
        for (int i = 0; i < arrDetection[0].size(); i++)
        {
            //std::cout<<"arrDetection object_class: "<<arrDetection[0][i].object_class<<std::endl;
            if (arrDetection[0][i].object_class == 0)
            {
                std::vector<DetectedObject>det(1, arrDetection[0][i]);
                // while(true)
                // {
               // double time0 = static_cast<double>(cv::getTickCount());
                alphaPose.getKeyPoints(img1, det, pts);
                //time0 = ((double)cv::getTickCount() - time0) / cv::getTickFrequency();
                //std::cout << "此方法运行时间 为:" << time0 *1000 << "毫秒" << std::endl;
                // }
                for (int i = 0; i < pts.size(); i++)
                {
                    std::vector<cv::Point> batchImgPts = pts[i];
                    for (auto& pt : batchImgPts)
                    {
                        cv::circle(imgclone, pt, 1, cvScalar(0, 0, 255), 2);
                    }
                    //cv::imwrite("/data/ld/project/mlu-pytorch-alphapose-master/res_" + std::to_string(i) + ".jpg", imgclone);
                }
                pts.clear();
            }

        }
        writer.write(imgclone);
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