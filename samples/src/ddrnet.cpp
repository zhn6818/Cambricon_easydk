#include "inference.h"

#include <glog/logging.h>
#include <sys/time.h>

fpnSegment::fpnSegment(const std::string &modelPath, const std::string &func_name, const int device_id):BaseInfer(modelPath,func_name,device_id)
{

}



void fpnSegment::getfeat(cv::Mat & Img,cv::Mat& feat)
{
    std::vector<cv::Mat> BatchImgs{Img};
    //assert(BatchImgs.size() > 0);
    resizeCvtColorMlu(BatchImgs);
    modelInfer.Run(mluInput,mluOutput);
    memOp.MemcpyOutputD2H(cpuOutput,mluOutput);
    float* res = (float*)cpuOutput[0];
    //setbuf(stdout, 0);
    // for(int i = 0; i < 100; i++)
    // {
    //     std::cout<<res[i]<<" ";
    // }
    cv::Mat tmp = cv::Mat(out_h,out_w,CV_32FC2,res);
    cv::Mat resizeImg;
    cv::resize(tmp,resizeImg,cv::Size(256,128),0,0,cv::INTER_LINEAR);
    feat = cv::Mat(128,256,CV_8UC1);
    cv::Mat channels[2];
    cv::split(resizeImg,channels);
    feat = channels[0] <= channels[1];
    //std::cout<<resizeImg.isContinuous()<<std::endl;
}
void fpnSegment::getfeatold(cv::Mat  &Img, cv::Mat &feat)
{
    std::vector<cv::Mat> vBatchImgs{Img};
    //assert(vBatchImgs.size() > 0);
    resizeCvtColorMlu(vBatchImgs);
    modelInfer.Run(mluInput,mluOutput);
    memOp.MemcpyOutputD2H(cpuOutput,mluOutput);
    cv::Mat img = cv::Mat(out_h, out_w, CV_32FC1,cpuOutput[0]);
    img.convertTo(feat, CV_8UC1, 255.0);
    //cv::resize(feat,feat,cv::Size(vBatchImgs[0].cols,vBatchImgs[0].rows));
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(feat,feat,element);
}

void fpnSegment::processFeat_test(const cv::Mat &feat , std::vector<cv::Point> pts  ,const cv::Size &srcImageSize , float smoke_thres, std::vector<std::vector<cv::Point>>& contours , bool& isSmoke , int&binary_pixels,std::vector<std::vector<cv::Point>>& contours_all,cv::Mat& dstImg) 
{
   isSmoke = false;
    if(feat.empty())
        return;
    cv::Mat img;
    feat.convertTo(img, CV_8UC1);

    cv::Point* root_points  = new cv::Point[pts.size()];
    for(int i=0;i<pts.size();i++)
    {
        root_points[i].x = pts[i].x*feat.cols/(float)srcImageSize.width;
        root_points[i].y = pts[i].y*feat.rows/(float)srcImageSize.height;
        // LOG(INFO)<<"pts:"<<root_points[i];
    }

    const cv::Point* ppt[1] = { root_points };
    int npt[] = { int(pts.size())};


    cv::Mat mask_ann, dst;
    img.copyTo(mask_ann);
    mask_ann.setTo(cv::Scalar::all(0));

    fillPoly(mask_ann, ppt, npt, 1, cv::Scalar(255));
    // LOG(INFO)<<img.size()<<" "<<img.type()<<" "<<mask_ann.type();
    img.copyTo(dst, mask_ann);

    //imwrite("feat_roi.jpg" , dst);
    //cvtColor(dst,dst,COLOR_BGR2GRAY);
    threshold(dst,dst , 133,255,cv::THRESH_BINARY) ;
    
    cv::Mat showimg = dst.clone();
    //cv::imwrite("/data/ld/project/instanceseg_new/fog_ddr_model/res.jpg",showimg);
    // cv::Mat showimgbgr;
    // cv::cvtColor(showimg,showimgbgr,cv::COLOR_GRAY2BGR);
    // cv::imwrite("/data5/ld/project/psenet/showimg.jpg",showimgbgr);
    // all contours
    vector<cv::Vec4i> hierarchy;  //
    std::vector<std::vector<cv::Point>> contours_tmp;
    cv::findContours(showimg, contours_tmp, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    float scale_x = (float)srcImageSize.width/(float)showimg.cols;
    float scale_y = (float)srcImageSize.height/(float)showimg.rows;
    // std::vector<std::vector<cv::Point>> contours_filter ;

    for (int i=0;i<contours_tmp.size();i++)
    {
        //filter small contours
        //double area = contourArea(contours_tmp[i]) / (feat.cols*feat.rows/(256*256));
        double area = contourArea(contours_tmp[i]) * 256.0 * 256.0 / (feat.cols*feat.rows);
        cv::Rect rect1= boundingRect(cv::Mat(contours_tmp[i]));
        float tmp = (float)rect1.height/(float)rect1.width;
        float hwratio= tmp>1?tmp:1/(float)tmp;
        // LOG(INFO)<<"area:"<<area<<" "<<contours_tmp[i][0]<<" hwratio:"<<hwratio;
        if(area>150&&hwratio<4)
        {
            std::vector<cv::Point> conts;
            for(int j=0;j<contours_tmp[i].size();j++)
            {
                conts.push_back(contours_tmp[i][j]);
            }
            contours.push_back(conts);
        }
    }
    // LOG(INFO)<<"contours_filter SIZE:"<<contours.size();

    cv::Mat dst_filter = cv::Mat::zeros(dst.size(), CV_8UC1);
    drawContours(dst_filter, contours, -1, cv::Scalar(255) ,cv::FILLED);
    dstImg = dst_filter;
    // Mat dst_filter;
    // dst.copyTo(dst_filter, mask_contour);

    // imwrite("dst.jpg" ,dst);
    // imwrite("dst_filter.jpg" ,dst_filter);


    // Mat dst_norm;
    // dst_filter.convertTo(dst_norm, CV_32FC1, 1.);
    delete [] root_points;

    binary_pixels = sum(dst_filter).val[0] * 256 * 256 /(feat.cols*feat.rows * 255.0);
    // LOG(INFO)<<"binary_pixels:"<<binary_pixels;
   // exit(0);
    // float smoke_percent = 100-100*(float)binary_pixels / (float)(256*256);
    LOG(INFO)<<"SMOKE THRES:"<<binary_pixels <<" "<<smoke_thres ;
    if(binary_pixels>smoke_thres)
    {
        isSmoke= true;

        for (int i=0;i<contours.size();i++)
        {
            for(int j=0;j<contours[i].size();j++)
            {
                contours[i][j].x*=scale_x;
                contours[i][j].y*=scale_y;
            }
            // LOG(INFO)<<"COUNTSSS:"<<contours[i].size();
        }
        for (int i=0;i<contours_tmp.size();i++)
        {
            std::vector<cv::Point> conts_final;
            for(int j=0;j<contours_tmp[i].size();j++)
            {
                conts_final.push_back(cv::Point(contours_tmp[i][j].x*scale_x+3,contours_tmp[i][j].y*scale_y+3));
            }
            contours_all.push_back(conts_final);
        }
        LOG(INFO)<<"SMOKE :"<<isSmoke <<" contours size: "<<contours.size();
    }
    else
    {
        isSmoke = false;
        contours.clear();
        LOG(INFO)<<"SMOKE :"<<isSmoke <<" contours size: "<<contours.size();
    }
}
