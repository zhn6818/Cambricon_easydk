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

#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cncv.h"
#include "cnpostproc.h"
#include "cnrt.h"
#include "device/mlu_context.h"
#include "easybang/resize_and_colorcvt.h"
#include "easycodec/easy_decode.h"
#include "easyinfer/easy_infer.h"
#include "easyinfer/mlu_memory_op.h"
#include "easyinfer/model_loader.h"
#include "easytrack/easy_track.h"

#include <json/reader.h>
#include <json/value.h>
#include <json/writer.h>

using std::vector;

typedef std::pair<std::string, float> Prediction;

#define SAFECALL(func, expect)                                                     \
  do {                                                                             \
    auto ret = (func);                                                             \
    if (ret != (expect)) {                                                         \
      std::cerr << "Call " << #func << "failed, error code: " << ret << std::endl; \
      abort();                                                                     \
    }                                                                              \
  } while (0)

#define CNRT_SAFECALL(func) SAFECALL(func, CNRT_RET_SUCCESS)
#define CNCV_SAFECALL(func) SAFECALL(func, CNCV_STATUS_SUCCESS)

class Detection {
public:
    Detection(const std::string &model_path, const std::string &func_name, const int device_id,int yoloType = 1);

    void Preprocess(const std::vector<cv::Mat> &imgs, int dst_w, int dst_h, void *output);

    void KeepAspectRatio(cncvRect *dst_roi, const cncvImageDescriptor &src, const cncvImageDescriptor &dst);

    void Detect(std::vector<cv::Mat> &preprocessedImages, std::vector<std::vector<DetectedObject>> &arrDetection,
                std::vector<cv::Size> &sizeDetect);

    cv::Size GetSize();

    ~Detection();

    int GetBatch();

private:
    edk::MluMemoryOp mem_op_;
    edk::EasyInfer infer_;
    edk::MluResizeConvertOp rc_op_;
    edk::MluContext env_;
    std::shared_ptr<edk::ModelLoader> model_{nullptr};
    std::unique_ptr<edk::CnPostproc> postproc_{nullptr};
    std::unique_ptr<edk::FeatureMatchTrack> tracker_{nullptr};
    int m_yoloType;
    int device;
    int net_w;
    int net_h;
    int net_n;
    int net_c;
    int out_n;
    int out_c;
    int out_w;
    int out_h;
    uint32_t batch_size;
    void *model_input[1];
    void *model_output[1];
    size_t model_input_size;
    size_t model_output_size;

    std::vector<cv::Size> paras;

    cncvHandle_t handle;
    cnrtQueue_t queue;

    cncvImageDescriptor src_desc;
    cncvImageDescriptor tmp_desc;
    cncvImageDescriptor dst_desc;

    cncvRect *src_rois;
    cncvRect *tmp_rois;

    uint32_t cpu_src_imgs_buff_size, dst_size;
    void **cpu_src_imgs;
    void **cpu_dst_imgs;
    void **cpu_tmp_imgs;

    void **mlu_input;
    void **mlu_output;
    void **tmp;

    void **cpu_output_{nullptr};
};

class Classifycation {
public:
    Classifycation(const std::string &model_path, const std::string &func_name, const std::string &names,
                   const int device_id, std::string jsonPath = "");

    ~Classifycation();

    void Preprocess(const std::vector<cv::Mat> &imgs, int dst_w, int dst_h, void *output);

    std::vector<std::vector<Prediction>> Classify(std::vector<cv::Mat> &vBatchImages, int N = 1);

    int GetBatch();

private:
    std::vector<std::string> labels;
    cncvImageDescriptor src_desc;
    cncvImageDescriptor tmp_desc;
    cncvImageDescriptor dst_desc;

    cncvHandle_t handle;
    cnrtQueue_t queue;

    cncvRect *src_rois;
    cncvRect *tmp_rois;
    void **cpu_src_imgs;
    void **cpu_tmp_imgs;
    void **cpu_dst_imgs;

    void **mlu_input;
    void **mlu_output;
    void **tmp;

    edk::MluMemoryOp mem_op_;
    edk::EasyInfer infer_;
    edk::MluResizeConvertOp rc_op_;
    edk::MluContext env_;
    std::shared_ptr<edk::ModelLoader> model_{nullptr};
    std::unique_ptr<edk::CnPostproc> postproc_{nullptr};

    uint32_t batch_size;
    uint32_t cpu_src_imgs_buff_size, dst_size;
    void *model_input[1];
    // void* model_output[1];
    void **model_output;

    size_t model_input_size;
    size_t model_output_size;

    int device;
    int net_w;
    int net_h;
    int net_n;
    int net_c;
    int out_n;
    int out_c;
    int out_w;
    int out_h;

    float shift;

    void **cpu_output_{nullptr};
};

// class DetectionRunner : public StreamRunner {
class DetectionRunner {
public:
    DetectionRunner(const std::string &model_path, const std::string &func_name, const int device_id);

    ~DetectionRunner();

    void Detect(std::vector<std::vector<cv::Mat>> &preprocessedImages,
                std::vector<std::vector<DetectedObject>> &arrDetection, std::vector<cv::Size> &image_paras);

    static void Pre(std::vector<cv::Mat> &vBatchImages, std::vector<std::vector<cv::Mat>> &preprocessedImages,
                    std::vector<cv::Size> &image_paras);

    int GetBatch();

private:
    static void WrapInputLayer(int batchsize, std::vector<std::vector<cv::Mat>> *wrappedImages);

    static void Preprocess(const std::vector<cv::Mat> &sourceImages, std::vector<std::vector<cv::Mat>> *destImages);

    edk::MluMemoryOp mem_op_;
    edk::EasyInfer infer_;
    edk::MluResizeConvertOp rc_op_;
    edk::MluContext env_;
    std::shared_ptr<edk::ModelLoader> model_{nullptr};
    std::unique_ptr<edk::CnPostproc> postproc_{nullptr};
    std::unique_ptr<edk::FeatureMatchTrack> tracker_{nullptr};
    void **mlu_output_{nullptr}, **cpu_output_{nullptr}, **mlu_input_{nullptr};

    int net_w;
    int net_h;
    int net_n;
    int net_c;
    int out_n;
    int out_c;
    int out_w;
    int out_h;

    void **input_data;  //输入数据，在cpu上的指针
    void **cpuData_;
    void **cpuTrans_;
    void **firstConvData_;
    void **inputMluPtrS_infer;
    void **inputMluTempPtrS_infer;
    void **mluData_infer;
    void **cpuTempData_infer;
};

class ClassificationRunner {
public:
    ClassificationRunner(const std::string &model_path, const std::string &func_name, const std::string &names,
                         const int device_id);

    ~ClassificationRunner();

    std::vector<std::vector<Prediction>> Classify(std::vector<cv::Mat> &vBatchImages, int N = 1);

    int GetBatch();

private:
    void Pre(const std::vector<cv::Mat> &vBatchImages);

    edk::MluMemoryOp mem_op_;
    edk::EasyInfer infer_;
    edk::MluResizeConvertOp rc_op_;
    edk::MluContext env_;
    std::shared_ptr<edk::ModelLoader> model_{nullptr};
    std::unique_ptr<edk::CnPostproc> postproc_{nullptr};
    void **mlu_output_{nullptr}, **cpu_output_{nullptr};
    std::vector<std::string> labels;

    int net_w;
    int net_h;
    int net_n;
    int net_c;
    int out_n;
    int out_c;
    int out_w;
    int out_h;
    void **inputCpuPtrS;
    void **mluData_infer;
};

class CrowdCountPredictor {
public:
    CrowdCountPredictor(const std::string &model_path, const std::string &func_name, const int device_id);

    ~CrowdCountPredictor();

    void run(cv::Mat &img, cv::Mat &out);

    int GetBatch();

private:
    edk::MluMemoryOp mem_op_;
    edk::EasyInfer infer_;
    edk::MluResizeConvertOp rc_op_;
    edk::MluContext env_;
    std::shared_ptr<edk::ModelLoader> model_{nullptr};
    void **mlu_output_{nullptr}, **cpuData_{nullptr}, **cpu_output_{nullptr}, **mlu_input_{nullptr};

    void Pre(cv::Mat &vBatchImages);

    // void WrapInputLayer(std::vector<cv::Mat>&wrappedImages);
    // void Preprocess(cv::Mat &img);
    // std::vector<cv::Mat>wrappedImages;
    int net_w;
    int net_h;
    int net_n;
    int net_c;
    int out_n;
    int out_c;
    int out_w;
    int out_h;
};

class ResnetSegment {
public:
    ResnetSegment(const std::string &model_path, const std::string &func_name, const int device_id);

    ~ResnetSegment();

    void Pre(const std::vector<cv::Mat> &vBatchImages);

    void getfeat(std::vector<cv::Mat> &vBatchImages, cv::Mat &feat);

    void processFeat_test(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size &srcImageSize,
                          float smoke_thresh, std::vector<std::vector<cv::Point>> &contours, bool &isSmoke,
                          int &binary_pixels, std::vector<std::vector<cv::Point>> &contours_all);

    void processFeat(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size &srcImageSize, float smoke_thres,
                     std::vector<std::vector<cv::Point>> &contours, bool &isSmoke);

private:
    edk::MluContext env_;
    edk::MluMemoryOp mem_op_;
    edk::EasyInfer infer_;
    std::shared_ptr<edk::ModelLoader> model_{nullptr};
    void **mlu_output_{nullptr}, **inputCpuPtrS{nullptr}, **cpu_output_{nullptr}, **mluData_infer{nullptr};
    int net_w;
    int net_h;
    int net_n;
    int net_c;
    int out_n;
    int out_c;
    int out_w;
    int out_h;
};

class Segment {
public:
    Segment(const std::string &model_path, const std::string &func_name, const int device_id);

    ~Segment();

    void Preprocess(const std::vector<cv::Mat> &imgs, int dst_w, int dst_h, void *output);

    void getfeat(std::vector<cv::Mat> &vBatchImgs, cv::Mat &feat);

    int GetBatch();

    void processFeat(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size &srcImageSize, float smoke_thres,
                     std::vector<std::vector<cv::Point>> &contours, bool &isSmoke);

    void processFeat_test(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size &srcImageSize,
                          float smoke_thres, std::vector<std::vector<cv::Point>> &contours, bool &isSmoke,
                          int &binary_pixels, std::vector<std::vector<cv::Point>> &contours_all);

private:
    std::vector<std::string> labels;
    cncvImageDescriptor src_desc;
    cncvImageDescriptor tmp_desc;
    cncvImageDescriptor dst_desc;

    cncvHandle_t handle;
    cnrtQueue_t queue;

    cncvRect *src_rois;
    cncvRect *tmp_rois;
    void **cpu_src_imgs;
    void **cpu_tmp_imgs;
    void **cpu_dst_imgs;
    void **mlu_input;
    void **mlu_output;
    void **tmp;

    edk::MluMemoryOp mem_op_;
    edk::EasyInfer infer_;
    edk::MluResizeConvertOp rc_op_;
    edk::MluContext env_;
    std::shared_ptr<edk::ModelLoader> model_{nullptr};
    std::unique_ptr<edk::CnPostproc> postproc_{nullptr};

    uint32_t batch_size;
    uint32_t cpu_src_imgs_buff_size, dst_size;
    void *model_input[1];
    void **model_output;
    size_t model_input_size;
    size_t model_output_size;

    int device;
    int net_w;
    int net_h;
    int net_n;
    int net_c;
    int out_n;
    int out_c;
    int out_w;
    int out_h;

    void **cpu_output_{nullptr};
};


// class OpencvMtcnn : public Mtcnn {
// public:
//     //int load_model(const std::vector<std::string> &pnet_model_path, std::string rnet_model_path,std::string onet_model_path, const std::string &func_name, int ngpuid);
//     int load_model(const std::string& modelfolder,const std::string &func_name, int ngpuid);
//     void detect(cv::Mat &img, std::vector<face_box> &face_list);
//     ~OpencvMtcnn();
// private:
//     static const int pnet_model_num = 10;

//     int run_PNet(const cv::Mat &img, int index, std::vector<BoundingBox> &box_list);

//     void run_RNet(const cv::Mat &img, std::vector<BoundingBox> &pnet_boxes, std::vector<BoundingBox> &output_boxes);

//     void run_ONet(const cv::Mat &img, std::vector<BoundingBox> &rnet_boxes, std::vector<BoundingBox> &output_boxes);
    
//     void nmsGlobal(std::vector<BoundingBox>& totalBoxes);
    
//     void nms(std::vector<BoundingBox> *boxes, float threshold,int type, std::vector<BoundingBox> *filterOutBoxes);

//     void generateBoundingBox(float* boxRegs,int i,float* cls,float scale_w, float scale_h,const float threshold,std::vector<BoundingBox> & filterOutBoxes);
    
//     void filteroutBoundingBox(const vector<BoundingBox> &boxes,const vector<float> &boxRegs,const vector<float> &cls, 
    
//     const vector<float> &points, const vector<int> &points_shape,float threshold, vector<BoundingBox> *filterOutBoxes);
//     std::vector<std::string> pnet_model_path;
//     std::string rnet_model_path;
//     std::string onet_model_path;

//     edk::EasyInfer pnet_model_infer[pnet_model_num];
//     edk::EasyInfer rnet_model_infer;
//     edk::EasyInfer onet_model_infer;

//     edk::MluResizeConvertOp pnet_rc_op[pnet_model_num];
//     edk::MluResizeConvertOp rnet_rc_op;
//     edk::MluResizeConvertOp onet_rc_op;

//     edk::MluContext pnet_env[pnet_model_num];
//     edk::MluContext rnet_env;
//     edk::MluContext onet_env;

//     edk::MluMemoryOp pnet_model_mem_op[pnet_model_num];
//     edk::MluMemoryOp rnet_model_mem_op;
//     edk::MluMemoryOp onet_model_mem_op;

//     std::shared_ptr<edk::ModelLoader> pnet_model[pnet_model_num]{nullptr};
//     std::shared_ptr<edk::ModelLoader> rnet_model{nullptr};
//     std::shared_ptr<edk::ModelLoader> onet_model{nullptr};


//     void **pnet_mlu_output[pnet_model_num]{nullptr}, **pnet_cpu_output[pnet_model_num]{
//             nullptr}, **pnet_mlu_input[pnet_model_num]{nullptr}, **pnet_cpu_input[pnet_model_num]{nullptr},*pnet_cpu_trans_input[pnet_model_num]{nullptr};
//     void **rnet_mlu_output{nullptr}, **rnet_cpu_output{nullptr}, **rnet_mlu_input{nullptr}, **rnet_cpu_input{nullptr};
//     void **onet_mlu_output{nullptr}, **onet_cpu_output{nullptr}, **onet_mlu_input{nullptr}, **onet_cpu_input{nullptr};
//     float *reg_cout_trans[pnet_model_num]{nullptr},*confidence_trans[pnet_model_num]{nullptr};
//     float *rnet_reg_cout_trans{nullptr},*rnet_confidence_trans{nullptr};
//     float *onet_reg_cout_trans{nullptr},*onet_confidence_trans{nullptr},*onet_landmark_trans{nullptr};
//     void *rnet_input_trans{nullptr},*onet_input_trans{nullptr};

//     int pnet_w[pnet_model_num];
//     int pnet_h[pnet_model_num];
//     int pnet_n[pnet_model_num];
//     int pnet_c[pnet_model_num];
//     int reg_n[pnet_model_num];
//     int reg_c[pnet_model_num];
//     int reg_w[pnet_model_num];
//     int reg_h[pnet_model_num];

//     int rnet_w;
//     int rnet_h;
//     int rnet_n;
//     int rnet_c;
//     int rnet_reg_w;
//     int rnet_reg_h;
//     int rnet_reg_c;
//     int rnet_reg_n;

//     int onet_w;
//     int onet_h;
//     int onet_c;
//     int onet_n;
//     int onet_reg_w;
//     int onet_reg_h;
//     int onet_reg_c;
//     int onet_reg_n;
//     int landmark_n;
//     int landmark_c;
//     int landmark_h;
//     int landmark_w;
// private:
//     std::vector<BoundingBox> total_pnet_boxes;
//     std::vector<BoundingBox> total_rnet_boxes;
//     std::vector<BoundingBox> total_onet_boxes;
//     float scale;
//     int dx,dy;

// };


struct MatAndRois
{
    cv::Mat input;
    cv::Size sizeInput;
    std::vector<cv::Mat> matRois;
    std::vector<cv::Rect> rectRois;
};
struct BBox
{
    float x1, y1, x2, y2;
};

struct BBoxInfo
{
    BBox box;
    int label;
    float prob;
};
class HDDetectYolov3Private;
class HDDetectYolov3
{
  public:
    cv::Size GetSize();
    HDDetectYolov3(std::string weights, std::string func_name, int yoloType = 1,int gpuid = 0, float thresh = 0.5);
    int GetBatch();
    void adjustRect(std::vector<cv::Rect> &rects, cv::Size sizeInput);
    void detect2(std::vector<MatAndRois> &matRoiInputs, std::vector<std::vector<DetectedObject>> &res);
    void detect(std::vector<cv::Mat> &img, std::vector<std::vector<DetectedObject>> &res, std::vector<cv::Rect> vecRoi = std::vector<cv::Rect>(), int area = 2500);
  private:
    std::shared_ptr<HDDetectYolov3Private> m_pHandleHDDetectYolov3Private;
};

class BaseInfer
{
    public:
        void** cpuInput{nullptr},**mluInput{nullptr},**cpuOutput{nullptr},**mluOutput{nullptr};
        int net_n,net_c,net_h,net_w,out_n,out_c,out_h,out_w;
        edk::EasyInfer modelInfer;
        edk::MluMemoryOp memOp;
        edk::MluContext context;
        edk::MluResizeConvertOp resizeOp;
        std::shared_ptr<edk::ModelLoader> modelLoader{nullptr};
        void **cpu_src_imgs;
        void **cpu_tmp_imgs;
        void **cpu_dst_imgs;
        void **mlu_src_input;
        void **mlu_tmp_input;
        void **mlu_dst_input;
        cncvImageDescriptor src_desc;
        cncvImageDescriptor tmp_desc;
        cncvImageDescriptor dst_desc;
        cncvHandle_t handle;
        cnrtQueue_t queue;
        cncvRect *src_rois;
        cncvRect *tmp_rois;
        void *workspace;
        size_t workspace_size;
        uint32_t cpu_src_imgs_buff_size;

    public:
        BaseInfer(const std::string &modelPath, const std::string &func_name, const int device_id);
        ~BaseInfer();
        void resizeCvtColorCpu(const std::vector<cv::Mat>& imgs);
        void resizeCvtColorMlu(const std::vector<cv::Mat>& imgs);
        int GetBatch();
};

class fpnSegment : public BaseInfer
{
    public:
        fpnSegment(const std::string &modelPath, const std::string &func_name, const int device_id);
        void getfeat(cv::Mat &image, cv::Mat &feat);
        void getfeatold(cv::Mat &image, cv::Mat &feat);
        void processFeat_test(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size &srcImageSize,
                            float smoke_thres, std::vector<std::vector<cv::Point>> &contours, bool &isSmoke,
                            int &binary_pixels, std::vector<std::vector<cv::Point>> &contours_all,cv::Mat& dstImg);
};

class AlphaPose : public BaseInfer
{
    public:
        AlphaPose(const std::string &modelPath, const std::string &func_name, const int device_id);
        void getKeyPoints(cv::Mat &BatchImgs,std::vector<DetectedObject> &arrDetection,std::vector<std::vector<cv::Point>>& pts);

};
