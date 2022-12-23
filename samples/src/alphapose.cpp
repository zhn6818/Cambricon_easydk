#include "inference.h"

#include <glog/logging.h>
#include <sys/time.h>
#include <math.h>
#define PI 3.141592
AlphaPose::AlphaPose(const std::string &modelPath, const std::string &func_name, const int device_id):BaseInfer(modelPath,func_name,device_id)
{

}
static int sign(float num)
{
    if(num > 0)
        return 1;
    else if(num == 0)
        return 0;
    else
        return -1;
}
//convert box coord to center and scale
static void _box_to_center_scale(float x, float y, float w, float h,cv::Point2f& center, std::pair<float,float>& scale)
{
    int pixle_std = 1;
    center.x = x + w * 0.5;
    center.y = y + h * 0.5;
    float aspect_ratio = 0.75;
    float scale_mult = 1.25;
    if( w > aspect_ratio * h)
    {
        h = w / aspect_ratio;
    }
    else if( w < aspect_ratio * h)
    {
        w = h * aspect_ratio;
    }
    scale = std::make_pair<float,float>(w * 1.0 / pixle_std, h * 1.0 / pixle_std);
    if(center.x != -1)
    {
        scale.first *= scale_mult;
        scale.second *= scale_mult;
    }
}
static void _center_scale_to_box(cv::Point2f& center,std::pair<float,float>& scale,BBox& box)
{
    float pixel_std = 1.0;
    float w = scale.first * pixel_std;
    float h = scale.second * pixel_std;
    box.x1 = center.x - w * 0.5;
    box.y1 = center.y - h * 0.5;
    box.x2 = center.x + w * 0.5;
    box.y2 = center.y + h * 0.5; 
}
static void get_dir(std::pair<float,float>& src_point,float rot_rad, std::pair<float,float>& src_result)
{
    double sn = sin(rot_rad);
    double cs = cos(rot_rad);
    src_result = std::make_pair<float,float>(src_point.first * cs - src_point.second * sn, src_point.first * sn + src_point.second * cs);
}

static void get_affine_transform(cv::Point2f& center,std::pair<float,float>scale,float rot,int net_w,int net_h,bool inv,cv::Mat& trans)
{
    float src_w = scale.first;
    float rot_rad = PI * rot / 180;
    std::pair<float,float> src_dir;
    std::pair<float,float> dst_dir{0,net_w * -0.5};
    std::pair<float,float> src_point{0,src_w * -0.5};
    get_dir(src_point,rot_rad,src_dir);
    cv::Point2f src[3],dst[3];
    src[0].x = center.x ;
    src[0].y = center.y ;
    src[1].x = center.x + src_dir.first;
    src[1].y = center.y + src_dir.second;
    float direct[2];
    direct[0] = src[0].x - src[1].x;
    direct[1] = src[0].y - src[1].y;
    src[2].x = src[1].x - direct[1];
    src[2].y = src[1].y + direct[0];
    dst[0].x = net_w * 0.5;
    dst[0].y = net_h * 0.5;
    dst[1].x = net_w * 0.5 + dst_dir.first;
    dst[1].y = net_h * 0.5 + dst_dir.second;
    direct[0] = dst[0].x - dst[1].x;
    direct[1] = dst[0].y - dst[1].y;
    dst[2].x = dst[1].x - direct[1];
    dst[2].y = dst[1].y + direct[0];
    if(inv)
    {
        trans = cv::getAffineTransform(dst,src);
    }
    else
    {
        trans = cv::getAffineTransform(src,dst);
    }
    
}


void AlphaPose::getKeyPoints(cv::Mat &img,std::vector<DetectedObject> &arrDetection,std::vector<std::vector<cv::Point>>& pts)
{
    //assert(arrDetection.size() > 0 && arrDetection.size() <= net_n && pts.size() == 0);
    std::vector<cv::Mat> dstVec;
    std::vector<BBox> boxes;
    cv::Point2f center;
    bool inv = false;
    cv::Mat trans;
    std::pair<float, float> scale;
    for(int i = 0; i < arrDetection.size(); i++)
    {
        cv::Rect bounding_box = arrDetection[i].bounding_box;
        _box_to_center_scale(bounding_box.x, bounding_box.y, bounding_box.width, bounding_box.height, center, scale);
        get_affine_transform(center, scale, 0, net_w, net_h, inv, trans);
        cv::Mat dst;
        cv::warpAffine(img, dst, trans, cv::Size(net_w, net_h), cv::INTER_LINEAR);
        //cv::imwrite("/data/ld/project/mlu-pytorch-alphapose-master/tmp.jpg",dst);
        BBox box;
        _center_scale_to_box(center, scale, box);
        dstVec.push_back(dst);
        boxes.push_back(box);
    }
    //std::vector<cv::Mat> dstVec{dst};
    resizeCvtColorCpu(dstVec);
    //resizeCvtColorMlu(dstVec);
    modelInfer.Run(mluInput,mluOutput);
    memOp.MemcpyOutputD2H(cpuOutput,mluOutput);
    void* cpuOutpuTrans = (void*)malloc(out_n*out_c*out_w*out_h*sizeof(float));
    int dim_shape[4] = {out_n,out_h,out_w,out_c};
    int dim_order[4] = {0,3,1,2};
    cnrtTransDataOrder(cpuOutput[0],CNRT_FLOAT32,cpuOutpuTrans,4,dim_shape,dim_order);
    // for(int i =0; i < 100; i++)
    // {
    //     LOG(INFO)<<((float*)(cpuOutpuTrans))[i]<<" ";
    // }
    for(int b = 0; b < arrDetection.size(); b++)
    {
        std::vector<cv::Point> batchPts;
        float* batch = (float*)(cpuOutpuTrans) + b * out_c * out_h * out_w;
        for(int k =0; k < out_c; k++)
        {
            float max = 0;
            float xMax = 0;
            float yMax = 0;
            float* begin = batch + k * out_h * out_w;
            for(int i = 0; i < out_h;i++)
            {
                for(int j =0; j < out_w;j++)
                {
                    float val = *(begin + i * out_w + j);
                    if(val > max)
                    {
                        max = val;
                        xMax = j;
                        yMax = i;
                    }
                }
            }
            yMax += sign(*(begin + int((yMax + 1) * out_w + xMax)) - *(begin + int((yMax - 1) * out_w + xMax))) * 0.25;
            xMax += sign(*(begin +  int(yMax * out_w + xMax + 1)) - *(begin + int(yMax * out_w + xMax - 1))) * 0.25;
            center.x = (boxes[b].x1 + boxes[b].x2)*0.5;
            center.y = (boxes[b].y1 + boxes[b].y2)*0.5;
            scale.first = boxes[b].x2 - boxes[b].x1;
            scale.second = boxes[b].y2 - boxes[b].y1;
            inv = true;
            get_affine_transform(center,scale,0,out_w,out_h,inv,trans);  
            xMax = trans.at<double>(0,0) * xMax + trans.at<double>(0,1) * yMax + trans.at<double>(0,2);
            yMax = trans.at<double>(1,0) * xMax + trans.at<double>(1,1) * yMax + trans.at<double>(1,2);
            if(max > 0.0)
            {
                batchPts.push_back(cv::Point(xMax,yMax));
            }
        }
        pts.push_back(batchPts);
    }
    // for(int i = 0; i < pts.size(); i++)
    // {
    //     std::vector<cv::Point> batchImgPts = pts[i];
    //     for(auto& pt : batchImgPts)
    //     {
    //         cv::circle(img,pt,1, cvScalar(0, 0, 255), 2);
    //     }
    //     cv::imwrite("/data/ld/project/mlu-pytorch-alphapose-master/res_0.jpg",img);
    // }
    free(cpuOutpuTrans);
}
