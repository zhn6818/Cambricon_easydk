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

#include "inference.h"

#include <glog/logging.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>



#if CV_VERSION_EPOCH == 2
#define OPENCV_MAJOR_VERSION 2
#elif CV_VERSION_MAJOR >= 3
#define OPENCV_MAJOR_VERSION CV_VERSION_MAJOR
#endif

constexpr bool keep_aspect_ratio = false;

float prob_sigmoid(float x) { return (1 / (1 + exp(-x))); }

int Detection::GetBatch() { return net_n; }

cv::Size Detection::GetSize() { return cv::Size(net_w, net_h); }

void Detection::KeepAspectRatio(cncvRect *dst_roi, const cncvImageDescriptor &src, const cncvImageDescriptor &dst) {
    float src_ratio = static_cast<float>(src.width) / src.height;
    float dst_ratio = static_cast<float>(dst.width) / dst.height;
    if (src_ratio < dst_ratio) {
        int pad_lenth = dst.width - src_ratio * dst.height;
        pad_lenth = (pad_lenth % 2) ? pad_lenth - 1 : pad_lenth;
        if (dst.width - pad_lenth / 2 < 0)
            return;
        dst_roi->w = dst.width - pad_lenth;
        dst_roi->x = pad_lenth / 2;
        dst_roi->y = 0;
        dst_roi->h = dst.height;
    } else if (src_ratio > dst_ratio) {
        int pad_lenth = dst.height - dst.width / src_ratio;
        pad_lenth = (pad_lenth % 2) ? pad_lenth - 1 : pad_lenth;
        if (dst.height - pad_lenth / 2 < 0)
            return;
        dst_roi->h = dst.height - pad_lenth;
        dst_roi->y = pad_lenth / 2;
        dst_roi->x = 0;
        dst_roi->w = dst.width;
    }
}

void Detection::Preprocess(const std::vector<cv::Mat> &imgs, int dst_w, int dst_h, void *output) {
    if (imgs.empty()) {
        std::cerr << "no image" << std::endl;
        abort();
    }
    // CNCV算子的batchsize是运行时可变的，有多少数据就设成对应的batchsize可以节省算力
    //std::cout << "start preprocessing" << std::endl;
    assert(batch_size == imgs.size());
    // 下面的资源申请都可以放到初始化阶段做
    // 运行过程中不需要malloc/new/CreateQueue等操作
    // 图像规模不变的话desc和roi也可以固定下来
    // 模型batchsize是固定的，preproc处理数不会超过模型batchsize，所以初始化时可以直接按照模型batchsize申请资源
    // cncvHandle_t handle;
    // cnrtQueue_t queue;
    // // 不同线程不能共用queue和handle
    // CNRT_SAFECALL(cnrtCreateQueue(&queue));
    // CNCV_SAFECALL(cncvCreate(&handle));
    // CNCV_SAFECALL(cncvSetQueue(handle, queue));
    // cncvImageDescriptor src_desc;
    // cncvImageDescriptor tmp_desc;
    // cncvImageDescriptor dst_desc;
    // std::cout << "start preprocessing" << std::endl;
    uint32_t maxSize = 0;
    for (int i = 0; i < imgs.size(); ++i) {
        uint32_t tmpSize = imgs[i].rows * imgs[i].step * sizeof(uint8_t);
        if (tmpSize > maxSize) {
            maxSize = tmpSize;
        }
    }

    cncvRect dst_roi;

    void *workspace;
    size_t workspace_size;

    int src_w = imgs[0].cols;
    int src_h = imgs[0].rows;
    int src_stride = imgs[0].step;

    //std::cout << "maxSize: " << maxSize << std::endl;
    //assert(maxSize <= cpu_src_imgs_buff_size);
    while (maxSize > cpu_src_imgs_buff_size) {
        for (uint32_t i = 0; i < batch_size; ++i) {
            cnrtFree(cpu_src_imgs[i]);
            std::cout << "free histroy mlu memry" << std::endl;
        }
        cpu_src_imgs_buff_size = maxSize;
        for (uint32_t idx = 0; idx < batch_size; ++idx) {
            CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
            std::cout << "remalloc mlu memory: " << cpu_src_imgs_buff_size << std::endl;
        }
    }
    uint32_t src_size;
    // copy src imgs to mlu
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        // CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), src_size));
        src_size = src_h * src_stride * sizeof(uint8_t);
        // std::cout << idx << " step :" << imgs[idx].rows * imgs[idx].step * sizeof(uint8_t) << std::endl;
        CNRT_SAFECALL(cnrtMemcpy(cpu_src_imgs[idx], imgs[idx].data, src_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    }

    // copy mlu src imgs pointer cpu array to mlu
    CNRT_SAFECALL(cnrtMemcpy(mlu_input, cpu_src_imgs, batch_size * sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV));

    CNRT_SAFECALL(cnrtMemcpy(tmp, cpu_tmp_imgs, batch_size * sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV));

    // wrap output memory into dst pointer array
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        cpu_dst_imgs[idx] = reinterpret_cast<uint8_t *>(output) + idx * dst_size;
    }
    // copy mlu dst imgs pointer cpu array to mlu
    //   CNRT_SAFECALL(cnrtMalloc((void**)&mlu_output, batch_size * sizeof(void*)));
    CNRT_SAFECALL(cnrtMemcpy(mlu_output, cpu_dst_imgs, batch_size * sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV));

    // batchsize越大，需要的workspace_size也会越大，每次运行都需要检查是否够大，不满足需求需要刷新
    // size_t tmp = workspace_size;
    CNCV_SAFECALL(cncvGetResizeRgbxWorkspaceSize(batch_size, &workspace_size));
    // std::cout << "workspace_size: " << workspace_size << std::endl;
    // if(tmp < workspace_size)
    // {
    //   cnrtFree(workspace);
    CNRT_SAFECALL(cnrtMalloc(&workspace, workspace_size));
    // }

    src_desc.width = src_w;
    src_desc.height = src_h;
    src_desc.pixel_fmt = CNCV_PIX_FMT_BGR;
    src_desc.stride[0] = src_stride;
    src_desc.depth = CNCV_DEPTH_8U;

    tmp_desc.width = dst_w;
    tmp_desc.height = dst_h;
    tmp_desc.pixel_fmt = CNCV_PIX_FMT_BGR;
    tmp_desc.stride[0] = dst_w * 3 * sizeof(uint8_t);
    tmp_desc.depth = CNCV_DEPTH_8U;

    for (uint32_t i = 0; i < batch_size; ++i) {
        // init dst rect
        tmp_rois[i].x = 0;
        tmp_rois[i].y = 0;
        tmp_rois[i].w = dst_w;
        tmp_rois[i].h = dst_h;

        // init src rect
        src_rois[i].x = 0;
        src_rois[i].y = 0;
        src_rois[i].w = src_w;
        src_rois[i].h = src_h;
        if (keep_aspect_ratio) {
            KeepAspectRatio(&tmp_rois[i], src_desc, tmp_desc);
        }
    }
    // async
    CNCV_SAFECALL(cncvResizeRgbx(handle, batch_size, src_desc, src_rois, mlu_input, tmp_desc, tmp_rois, tmp,
                                 workspace_size, workspace, CNCV_INTER_BILINEAR));

    dst_roi = tmp_rois[0];
    dst_desc = tmp_desc;
    dst_desc.pixel_fmt = CNCV_PIX_FMT_RGBA;
    dst_desc.stride[0] = dst_w * 4 * sizeof(uint8_t);
    // std::cout << "async" << std::endl;
    CNCV_SAFECALL(cncvRgbxToRgbx(handle, batch_size, tmp_desc, dst_roi, tmp, dst_desc, dst_roi, mlu_output));

    // wait for task finished
    CNRT_SAFECALL(cnrtSyncQueue(queue));

    cnrtFree(workspace);
}

Detection::~Detection() {
    cncvDestroy(handle);
    cnrtDestroyQueue(queue);
    for (uint32_t i = 0; i < batch_size; ++i) {
        cnrtFree(cpu_src_imgs[i]);
        cnrtFree(cpu_tmp_imgs[i]);
    }
    cnrtFree(mlu_input);
    cnrtFree(tmp);
    cnrtFree(mlu_output);
    // cnrtFree(workspace);
    delete[] cpu_src_imgs;
    delete[] cpu_tmp_imgs;
    delete[] cpu_dst_imgs;
    delete[] src_rois;
    delete[] tmp_rois;
}

Detection::Detection(const std::string &model_path, const std::string &func_name, const int device_id,int yoloType) {
    device = device_id;
    // initial environment
    std::cout << "initial Detection !" << std::endl;
    env_.SetDeviceId(device_id);
    std::cout << "setDevice success" << std::endl;
    env_.BindDevice();
    std::cout << "BindDevice success" << std::endl;
    // load offline model
    std::cout << "initial model start !" << std::endl;
    model_ = std::make_shared<edk::ModelLoader>(model_path.c_str(), func_name.c_str());
    std::cout << "initial model success !" << std::endl;
    // prepare mlu memory operator and memory
    mem_op_.SetModel(model_);
    // init easy_infer
    infer_.Init(model_, device_id);

    // create mlu resize and convert operator
    auto &in_shape = model_->InputShape(0);
    auto &out_shape = model_->OutputShape(0);
    int outNum = (int) model_->OutputNum();
    m_yoloType = yoloType;
    net_w = in_shape.W();
    net_h = in_shape.H();
    net_n = in_shape.N();
    net_c = in_shape.C();

    out_n = out_shape.N();
    out_c = out_shape.C();
    out_w = out_shape.W();
    out_h = out_shape.H();

    batch_size = net_n;

    model_input_size = net_h * net_w * net_c * net_n * sizeof(uint8_t);
    model_output_size = out_h * out_w * out_c * out_n * sizeof(int16_t);
    std::cout << "net_n:" << net_n << " net_c:" << net_c << " net_h:" << net_h << " net_w:" << net_w << std::endl;
    std::cout << "out_n:" << out_n << " out_c:" << out_c << " out_h:" << out_h << " out_w:" << out_w << std::endl;
    std::cout << "out numbers: " << outNum << std::endl;

    if(m_yoloType == 0)
    {
        postproc_.reset(new edk::Yolov3Postproc);
    }
    else if(m_yoloType == 1)
    {
        postproc_.reset(new edk::Yolov4Postproc);
    }
    postproc_->set_threshold(0.45);
    CHECK(postproc_);

    std::cout << "batch size: " << batch_size << std::endl;
    CNRT_SAFECALL(cnrtMalloc(&(model_input[0]), model_input_size));
    CNRT_SAFECALL(cnrtMalloc(&(model_output[0]), model_output_size));

    CNRT_SAFECALL(cnrtCreateQueue(&queue));
    CNCV_SAFECALL(cncvCreate(&handle));
    CNCV_SAFECALL(cncvSetQueue(handle, queue));

    src_rois = new cncvRect[batch_size];
    tmp_rois = new cncvRect[batch_size];

    cpu_src_imgs = new void *[batch_size];

    cpu_src_imgs_buff_size = 1 * sizeof(uint8_t);
    std::cout << "cpu_src_imgs_buff_size: " << cpu_src_imgs_buff_size << std::endl;

    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
    }

    CNRT_SAFECALL(cnrtMalloc((void **) &mlu_input, batch_size * sizeof(void *)));

    cpu_tmp_imgs = new void *[batch_size];

    dst_size = net_w * net_h * 4 * sizeof(uint8_t);
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        CNRT_SAFECALL(cnrtMalloc(&(cpu_tmp_imgs[idx]), dst_size));
    }

    CNRT_SAFECALL(cnrtMalloc((void **) &tmp, batch_size * sizeof(void *)));

    cpu_dst_imgs = new void *[batch_size];
    CNRT_SAFECALL(cnrtMalloc((void **) &mlu_output, batch_size * sizeof(void *)));

    cpu_output_ = mem_op_.AllocCpuOutput();
    // workspace_size = 0;

    std::cout << "initialize success" << std::endl;
}

void Detection::Detect(std::vector<cv::Mat> &preprocessedImages, std::vector<std::vector<DetectedObject>> &arrDetection,
                       std::vector<cv::Size> &sizeDetect) {
    // env_.SetDeviceId(device);
    //std::cout << "test";
    // paras.resize(0);
    env_.BindDevice();
    //std::cout << "BindDevice " << std::endl;
    // for(int i = 0; i < batch_size; i++)
    // {
    //   std::string sss = "/data1/zhn/suguan/testImg/oo" + std::to_string(i) + ".png";
    //   cv::imwrite(sss, preprocessedImages[i]);
    // }
    Preprocess(preprocessedImages, net_w, net_h, model_input[0]);

    // testInput(model_input[0]);
    // void **cpu_input;
    // cpu_input = mem_op_.AllocCpuInput();
    // mem_op_.MemcpyD2H(cpu_input[0], model_input[0], model_input_size);
    // std::cout << "uint8_t: " << sizeof(uint8_t) << std::endl;

    // cv::Mat img = cv::Mat(416 * 4, 416, CV_8UC4, cv::Scalar::all(0));

    // memcpy(img.data, (reinterpret_cast<uint8_t*>(cpu_input[0])), 416 * 416 * 4 * 4);
    // cv::Mat imgout;
    // cv::cvtColor(img, imgout, cv::COLOR_BGRA2BGR);
    // cv::imwrite("/data1/zhn/ttt.png", imgout);

    infer_.Run(model_input, model_output);
    // std::cout << "~~~~~~~~~~~run over~~~~~~~~~~~~~~~~" << std::endl;

    // // post process

    mem_op_.MemcpyOutputD2H(cpu_output_, (void **) model_output);
    std::vector<std::vector<edk::DetectObject>> objDetection;
    objDetection.clear();
    std::vector<std::pair<float *, uint64_t>> postproc_param;
    postproc_param.push_back(
            std::make_pair(reinterpret_cast<float *>(cpu_output_[0]), model_->OutputShape(0).BatchDataCount()));
    objDetection = postproc_->Execute(postproc_param, net_n);

    arrDetection.clear();

    for (int j = 0; j < net_n; j++) {
        int image_width = sizeDetect[j].width;
        int image_height = sizeDetect[j].height;
        float scale = std::min(float(net_w) / image_width, float(net_w) / image_height);
        float dx = (net_w - scale * image_width) / 2;
        float dy = (net_w - scale * image_height) / 2;
        std::vector<DetectedObject> objs;
        int len = objDetection[j].size();
        for (int i = 0; i < len; i++) {
            DetectedObject obj;
            float x0 = objDetection[j][i].bbox.x ;
            float y0 = objDetection[j][i].bbox.y ;
            float x1 = (objDetection[j][i].bbox.x + objDetection[j][i].bbox.width) ;
            float y1 = (objDetection[j][i].bbox.y + objDetection[j][i].bbox.height) ;
            if(m_yoloType == 0)
            {
                x0 *= net_w;
                y0 *= net_h;
                x1 *= net_w;
                y1 *= net_h;
            }
            x0 = (x0 - dx) / scale;
            y0 = (y0 - dy) / scale;
            x1 = (x1 - dx) / scale;
            y1 = (y1 - dy) / scale;
            x0 = (x0 > 0) ? x0 : 0;
            y0 = (y0 > 0) ? y0 : 0;
            x1 = (x1 < image_width) ? x1 : image_width - 1;
            y1 = (y1 < image_height) ? y1 : image_height - 1;
            objDetection[j][i].bbox.x = x0;
            objDetection[j][i].bbox.y = y0;
            objDetection[j][i].bbox.width = x1 - x0;
            objDetection[j][i].bbox.height = y1 - y0;
            if(objDetection[j][i].bbox.width <=0 || objDetection[j][i].bbox.height <= 0)
            {
                continue;
            }
            obj.object_class = objDetection[j][i].label;
            obj.prob = objDetection[j][i].score;
            obj.bounding_box.x = objDetection[j][i].bbox.x;
            obj.bounding_box.y = objDetection[j][i].bbox.y;
            obj.bounding_box.width = objDetection[j][i].bbox.width;
            obj.bounding_box.height = objDetection[j][i].bbox.height;
            objs.push_back(obj);
        }
        arrDetection.push_back(objs);
    }
}

Classifycation::~Classifycation() {
    if (nullptr != model_output)
        mem_op_.FreeMluOutput(model_output);
    cncvDestroy(handle);
    cnrtDestroyQueue(queue);
    for (uint32_t i = 0; i < batch_size; ++i) {
        cnrtFree(cpu_src_imgs[i]);
        cnrtFree(cpu_tmp_imgs[i]);
    }
    cnrtFree(mlu_input);
    cnrtFree(tmp);
    cnrtFree(mlu_output);
    // cnrtFree(workspace);
    delete[] cpu_src_imgs;
    delete[] cpu_tmp_imgs;
    delete[] cpu_dst_imgs;
    delete[] src_rois;
    delete[] tmp_rois;
}

int Classifycation::GetBatch() { return net_n; }

Classifycation::Classifycation(const std::string &model_path, const std::string &func_name, const std::string &names,
                               const int device_id, std::string jsonPath) {
    device = device_id;
    std::cout << "initial Classifycation !" << std::endl;
    env_.SetDeviceId(device_id);
    std::cout << "setDevice success" << std::endl;
    env_.BindDevice();
    std::cout << "BindDevice success" << std::endl;
    model_ = std::make_shared<edk::ModelLoader>(model_path.c_str(), func_name.c_str());
    std::cout << "initial model success !" << std::endl;
    // prepare mlu memory operator and memory
    mem_op_.SetModel(model_);
    // init easy_infer
    infer_.Init(model_, device_id);

    // create mlu resize and convert operator
    auto &in_shape = model_->InputShape(0);
    auto &out_shape = model_->OutputShape(0);
    int outNum = (int) model_->OutputNum();

    net_w = in_shape.W();
    net_h = in_shape.H();
    net_n = in_shape.N();
    net_c = in_shape.C();

    out_n = out_shape.N();
    out_c = out_shape.C();
    out_w = out_shape.W();
    out_h = out_shape.H();

    batch_size = net_n;

    model_input_size = net_h * net_w * net_c * net_n * sizeof(uint8_t);
    // model_output_size = out_h * out_w * out_c * out_n * sizeof(int32_t);
    // model_output_size = out_h * out_w * out_c * out_n * sizeof(uint8_t);

    std::cout << "net_n:" << net_n << " net_c:" << net_c << " net_h:" << net_h << " net_w:" << net_w << std::endl;
    std::cout << "out_n:" << out_n << " out_c:" << out_c << " out_h:" << out_h << " out_w:" << out_w << std::endl;
    std::cout << "out numbers: " << outNum << std::endl;

    std::cout<<"-----------------------------------------------------------------------++++++++++++++++++++++++++++++++++++------------------------------------------"<<names<<std::endl;
    std::ifstream fin(names, std::ios::in);
    char line[1024] = {0};
    std::string name = "";
    while (fin.getline(line, sizeof(line))) {
        std::stringstream word(line);
        word >> name;
        std::cout<< "--------------++++++++++++++++++++++++++++++++++++-------------"<<"name: " << name << std::endl;
        labels.push_back(name);
    }
    fin.clear();
    fin.close();

    postproc_.reset(new edk::ClassificationPostproc);
    postproc_->set_threshold(0.2);
    CHECK(postproc_);

    std::cout << "batch size: " << batch_size << std::endl;

    CNRT_SAFECALL(cnrtMalloc(&(model_input[0]), model_input_size));
    // CNRT_SAFECALL(cnrtMalloc(&(model_output[0]), model_output_size));
    // model_output.AllocMluOutput()
    model_output = mem_op_.AllocMluOutput();

    CNRT_SAFECALL(cnrtCreateQueue(&queue));
    CNCV_SAFECALL(cncvCreate(&handle));
    CNCV_SAFECALL(cncvSetQueue(handle, queue));

    src_rois = new cncvRect[batch_size];
    tmp_rois = new cncvRect[batch_size];

    CNRT_SAFECALL(cnrtMalloc((void **) &mlu_input, batch_size * sizeof(void *)));
    CNRT_SAFECALL(cnrtMalloc((void **) &tmp, batch_size * sizeof(void *)));
    CNRT_SAFECALL(cnrtMalloc((void **) &mlu_output, batch_size * sizeof(void *)));

    cpu_src_imgs = new void *[batch_size];
    cpu_src_imgs_buff_size = 100 * sizeof(uint8_t);
    std::cout << "cpu_src_imgs_buff_size: " << cpu_src_imgs_buff_size << std::endl;
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
    }

    cpu_tmp_imgs = new void *[batch_size];

    dst_size = net_w * net_h * 4 * sizeof(uint8_t);
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        CNRT_SAFECALL(cnrtMalloc(&(cpu_tmp_imgs[idx]), dst_size));
    }

    cpu_dst_imgs = new void *[batch_size];

    cpu_output_ = mem_op_.AllocCpuOutput();

    std::cout << "initialize Classifycation success" << std::endl;
    shift = 0;
    if(jsonPath.size() <= 0)
    {
      shift = 0;
      std::cout << "shift: " << shift << std::endl;
    }
    else{
      Json::Value root;
      Json::Reader reader;
      std::ifstream is(jsonPath);
      if (!is.is_open())
      {
        LOG(INFO) << "****open json file failed.***";
      }
      reader.parse(is, root);
      shift = root["shift"].asFloat();
      std::cout << "shift: " << shift << std::endl;
    }
}

std::vector<std::vector<Prediction>> Classifycation::Classify(std::vector<cv::Mat> &vBatchImages, int N) {
    //std::cout << "test classify" << std::endl;
    env_.BindDevice();
    Preprocess(vBatchImages, net_w, net_h, model_input[0]);
    //std::cout << "prePare over" << std::endl;
    infer_.Run(model_input, model_output);
    //std::cout << "Run over" << std::endl;
    mem_op_.MemcpyOutputD2H(cpu_output_, model_output);
    float* a = reinterpret_cast<float*>(cpu_output_[0]);
    if(shift != 0)
    {
      a[0] = a[0] + shift;
    }
    // for(int i = 0; i < out_c; i++)
    // {
    //   std::cout << *(a + i) << " ";
    // }
    //std::cout << "mem_op_ over" << std::endl;
    std::vector<std::vector<edk::DetectObject>> detect_result;
    std::vector<std::pair<float *, uint64_t>> postproc_param;
    postproc_param.push_back(
            std::make_pair(reinterpret_cast<float *>(cpu_output_[0]), model_->OutputShape(0).DataCount()));
    detect_result = postproc_->Execute(postproc_param, net_n);
    std::vector<std::vector<Prediction>> result;

    assert(detect_result.size() == net_n);
   
    for (int j = 0; j < net_n; ++j) {
        std::vector<Prediction> p;
        int len_result = detect_result[j].size();
        edk::DetectObject max_result = detect_result[j][0];
        for (int i = 0; i < len_result; i++) {
            if (max_result.score < detect_result[j][i].score) {
                max_result = detect_result[j][i];
            }
        }
        p.push_back(std::make_pair(labels[max_result.label], prob_sigmoid(max_result.score)));
        result.push_back(p);
    }
    return result;
}

void Classifycation::Preprocess(const std::vector<cv::Mat> &imgs, int dst_w, int dst_h, void *output) {
    if (imgs.empty()) {
        std::cerr << "no image" << std::endl;
        abort();
    }
    // std::cout << "start preprocessing" << std::endl;
    assert(batch_size == imgs.size());
    uint32_t maxSize = 0;
    for (int i = 0; i < imgs.size(); ++i) {
        uint32_t tmpSize = imgs[i].rows * imgs[i].step * sizeof(uint8_t);
        if (tmpSize > maxSize) {
            maxSize = tmpSize;
        }
    }
    // std::cout << "max size: " << maxSize << std::endl;
    cncvRect dst_roi;

    void *workspace;
    size_t workspace_size;

    int src_w = imgs[0].cols;
    int src_h = imgs[0].rows;
    int src_stride = imgs[0].step;
    // std::cout << " src_w: " << src_w << " src_h: " << src_h << " src_stride: " << src_stride << std::endl;

    while (maxSize > cpu_src_imgs_buff_size) {
        for (uint32_t i = 0; i < batch_size; ++i) {
            cnrtFree(cpu_src_imgs[i]);
            std::cout << "free histroy mlu memry" << std::endl;
        }
        cpu_src_imgs_buff_size = maxSize + 128;
        for (uint32_t idx = 0; idx < batch_size; ++idx) {
            CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
            std::cout << "remalloc mlu memory" << std::endl;
        }
    }
    uint32_t src_size;
    // copy src imgs to mlu
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        // CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), src_size));
        src_size = src_h * src_stride * sizeof(uint8_t);
        CNRT_SAFECALL(cnrtMemcpy(cpu_src_imgs[idx], imgs[idx].data, src_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    }

    CNRT_SAFECALL(cnrtMemcpy(mlu_input, cpu_src_imgs, batch_size * sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV));

    CNRT_SAFECALL(cnrtMemcpy(tmp, cpu_tmp_imgs, batch_size * sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV));
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        cpu_dst_imgs[idx] = reinterpret_cast<uint8_t *>(output) + idx * dst_size;
    }
    CNRT_SAFECALL(cnrtMemcpy(mlu_output, cpu_dst_imgs, batch_size * sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV));

    CNCV_SAFECALL(cncvGetResizeRgbxWorkspaceSize(batch_size, &workspace_size));
    CNRT_SAFECALL(cnrtMalloc(&workspace, workspace_size));

    src_desc.width = src_w;
    src_desc.height = src_h;
    src_desc.pixel_fmt = CNCV_PIX_FMT_BGR;
    src_desc.stride[0] = src_stride;
    src_desc.depth = CNCV_DEPTH_8U;

    tmp_desc.width = dst_w;
    tmp_desc.height = dst_h;
    tmp_desc.pixel_fmt = CNCV_PIX_FMT_BGR;
    tmp_desc.stride[0] = dst_w * 3 * sizeof(uint8_t);
    tmp_desc.depth = CNCV_DEPTH_8U;

    for (uint32_t i = 0; i < batch_size; ++i) {
        // init dst rect
        tmp_rois[i].x = 0;
        tmp_rois[i].y = 0;
        tmp_rois[i].w = dst_w;
        tmp_rois[i].h = dst_h;

        // init src rect
        src_rois[i].x = 0;
        src_rois[i].y = 0;
        src_rois[i].w = src_w;
        src_rois[i].h = src_h;
        // if (keep_aspect_ratio) {
        //   KeepAspectRatio(&tmp_rois[i], src_desc, tmp_desc);
        // }
    }
    CNCV_SAFECALL(cncvResizeRgbx(handle, batch_size, src_desc, src_rois, mlu_input, tmp_desc, tmp_rois, tmp,
                                 workspace_size, workspace, CNCV_INTER_BILINEAR));
    dst_roi = tmp_rois[0];
    dst_desc = tmp_desc;
    dst_desc.pixel_fmt = CNCV_PIX_FMT_BGRA;
    dst_desc.stride[0] = dst_w * 4 * sizeof(uint8_t);
    // std::cout << "async" << std::endl;
    CNCV_SAFECALL(cncvRgbxToRgbx(handle, batch_size, tmp_desc, dst_roi, tmp, dst_desc, dst_roi, mlu_output));

    // wait for task finished
    CNRT_SAFECALL(cnrtSyncQueue(queue));
    cnrtFree(workspace);
}

DetectionRunner::DetectionRunner(const std::string &model_path, const std::string &func_name, const int device_id) {
    // set mlu environment
    std::cout << "DetectionRunner" << std::endl;
    env_.SetDeviceId(device_id);
    env_.BindDevice();

    // load offline model
    model_ = std::make_shared<edk::ModelLoader>(model_path.c_str(), func_name.c_str());
    // prepare mlu memory operator and memory
    mem_op_.SetModel(model_);
    // init easy_infer
    infer_.Init(model_, device_id);

    // create mlu resize and convert operator
    auto &in_shape = model_->InputShape(0);
    auto &out_shape = model_->OutputShape(0);
    int outNum = (int) model_->OutputNum();

    net_w = in_shape.W();
    net_h = in_shape.H();
    net_n = in_shape.N();
    net_c = in_shape.C();

    out_n = out_shape.N();
    out_c = out_shape.C();
    out_w = out_shape.W();
    out_h = out_shape.H();

    std::cout << "net_n:" << net_n << " net_c:" << net_c << " net_h:" << net_h << " net_w:" << net_w << std::endl;
    std::cout << "out_n:" << out_n << " out_c:" << out_c << " out_h:" << out_h << " out_w:" << out_w << std::endl;
    std::cout << "out numbers: " << outNum << std::endl;

    postproc_.reset(new edk::Yolov3Postproc);
    postproc_->set_threshold(0.45);
    CHECK(postproc_);

    mlu_output_ = mem_op_.AllocMluOutput();
    cpu_output_ = mem_op_.AllocCpuOutput();
    inputMluTempPtrS_infer = reinterpret_cast<void **>(malloc(sizeof(void *) * 1));
    inputMluTempPtrS_infer[0] = malloc(net_n * net_c * net_w * net_h);
    inputMluPtrS_infer = reinterpret_cast<void **>(new void *[1]);
    cnrtMalloc(&(inputMluPtrS_infer[0]), net_n * net_c * net_w * net_h);
    cpuData_ = new (void *);
    cpuData_[0] = new float[net_n * 3 * net_w * net_h];
    cpuTrans_ = new (void *);
    cpuTrans_[0] = new float[net_n * 3 * net_w * net_h];
    firstConvData_ = new (void *);
    firstConvData_[0] = new char[net_n * 3 * net_w * net_h];
}

DetectionRunner::~DetectionRunner() {
    if (nullptr != mlu_output_)
        mem_op_.FreeMluOutput(mlu_output_);
    if (nullptr != cpu_output_)
        mem_op_.FreeCpuOutput(cpu_output_);

    if (nullptr != inputMluPtrS_infer)
        mem_op_.FreeMluInput(inputMluPtrS_infer);

    if (nullptr != inputMluTempPtrS_infer) {
        free(inputMluTempPtrS_infer[0]);
        free(inputMluTempPtrS_infer);
    }
    delete[] reinterpret_cast<float *>(cpuData_[0]);
    delete cpuData_;
    delete[] reinterpret_cast<float *>(cpuTrans_[0]);
    delete cpuTrans_;
    delete[] reinterpret_cast<char *>(firstConvData_[0]);
    delete firstConvData_;
}

int DetectionRunner::GetBatch() { return net_n; }

void DetectionRunner::WrapInputLayer(int batchsize, std::vector<std::vector<cv::Mat>> *wrappedImages) {
    int channels = 3;
    int h = 416;
    int w = 416;
    for (int i = 0; i < batchsize; ++i) {
        wrappedImages->push_back(std::vector<cv::Mat>());
        for (int j = 0; j < channels; ++j) {
            cv::Mat channel(h, w, CV_32FC1);
            (*wrappedImages)[i].push_back(channel);
        }
    }
}

void DetectionRunner::Preprocess(const std::vector<cv::Mat> &sourceImages,
                                 std::vector<std::vector<cv::Mat>> *destImages) {
    CHECK(sourceImages.size() == destImages->size()) << "Size of sourceImages and destImages doesn't match";

    int len_image = sourceImages.size();
    for (int i = 0; i < len_image; ++i) {
        cv::Mat sample;
        sample = sourceImages[i].clone();
        cv::Mat sample_temp;
        int input_dim = 416;
        cv::Mat sample_resized(input_dim, input_dim, CV_8UC3, cv::Scalar(128, 128, 128));
        cv::Size inGeometry_ = cv::Size(416, 416);
        if (sample.size() != inGeometry_) {
            float img_w = sample.cols;
            float img_h = sample.rows;
            int new_w = static_cast<int>(
                    img_w * std::min(static_cast<float>(input_dim) / img_w, static_cast<float>(input_dim) / img_h));
            int new_h = static_cast<int>(
                    img_h * std::min(static_cast<float>(input_dim) / img_w, static_cast<float>(input_dim) / img_h));
            cv::resize(sample, sample_temp, cv::Size(new_w, new_h), cv::INTER_CUBIC);
            sample_temp.copyTo(sample_resized(
                    cv::Range((static_cast<float>(input_dim) - new_h) / 2,
                              (static_cast<float>(input_dim) - new_h) / 2 + new_h),
                    cv::Range((static_cast<float>(input_dim) - new_w) / 2,
                              (static_cast<float>(input_dim) - new_w) / 2 + new_w)));
        }
        cv::Mat sample_float;
        sample_resized.convertTo(sample_float, CV_32FC3, 1);
        cv::split(sample_float, (*destImages)[i]);
        cv::Mat B_tmp = (*destImages)[i][0];
        (*destImages)[i][0] = (*destImages)[i][2];
        (*destImages)[i][2] = B_tmp;
    }
}

void DetectionRunner::Pre(std::vector<cv::Mat> &vBatchImages, std::vector<std::vector<cv::Mat>> &preprocessedImages,
                          std::vector<cv::Size> &image_paras) {
    int batchsize = vBatchImages.size();
    for (int i = 0; i < batchsize; i++) {
        cv::Size para(vBatchImages[i].cols, vBatchImages[i].rows);
        image_paras.push_back(para);
    }
    WrapInputLayer(batchsize, &preprocessedImages);
    Preprocess(vBatchImages, &preprocessedImages);
}

void DetectionRunner::Detect(std::vector<std::vector<cv::Mat>> &preprocessedImages,
                             std::vector<std::vector<DetectedObject>> &arrDetection,
                             std::vector<cv::Size> &image_paras) {
    float *data = reinterpret_cast<float *>(cpuData_[0]);
    int image_c = 3;
    int image_num = preprocessedImages.size();
    for (int i = 0; i < image_num; i++) {
        int channel_num = preprocessedImages[i].size();
        for (int j = 0; j < channel_num; j++) {
            memcpy(data, preprocessedImages[i][j].data, net_h * net_w * sizeof(float));
            data += net_w * net_h;
        }
    }

    mluData_infer = inputMluPtrS_infer;
    cpuTempData_infer = inputMluTempPtrS_infer;
    int inputNum = 1;
    for (int i = 0; i < inputNum; i++) {
        int dim_order[4] = {0, 2, 3, 1};
        int dim_shape[4] = {net_n, image_c, net_w, net_w};
        cnrtTransDataOrder(cpuData_[i], CNRT_FLOAT32, cpuTrans_[i], 4, dim_shape, dim_order);
        void *temp_ptr = nullptr;

        int input_count = net_n * image_c * net_w * net_h;

        cnrtCastDataType(cpuTrans_[i], CNRT_FLOAT32, firstConvData_[i], CNRT_UINT8, input_count, nullptr);

        int inputDimValue[4] = {net_n, net_w, net_w, image_c};
        int inputDimStride[4] = {0, 0, 0, 1};
        cnrtAddDataStride(firstConvData_[i], CNRT_UINT8, cpuTempData_infer[i], 4, inputDimValue, inputDimStride);

        temp_ptr = cpuTempData_infer[i];

        cnrtMemcpy(mluData_infer[i], temp_ptr, net_w * net_w * 4 * net_n, CNRT_MEM_TRANS_DIR_HOST2DEV);
    }
    infer_.Run(mluData_infer, mlu_output_);

    // post process
    mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_);
    std::vector<std::vector<edk::DetectObject>> objDetection;
    objDetection.clear();
    std::vector<std::pair<float *, uint64_t>> postproc_param;
    postproc_param.push_back(
            std::make_pair(reinterpret_cast<float *>(cpu_output_[0]), model_->OutputShape(0).BatchDataCount()));
    objDetection = postproc_->Execute(postproc_param, net_n);

    arrDetection.clear();
    for (int j = 0; j < net_n; j++) {
        int image_width = image_paras[j].width;
        int image_height = image_paras[j].height;
        float scale = std::min(float(net_w) / image_width, float(net_w) / image_height);
        float dx = (net_w - scale * image_width) / 2;
        float dy = (net_w - scale * image_height) / 2;
        std::vector<DetectedObject> objs;
        int len = objDetection[j].size();
        for (int i = 0; i < len; i++) {
            DetectedObject obj;
            float x0 = objDetection[j][i].bbox.x * net_w;
            float y0 = objDetection[j][i].bbox.y * net_w;
            float x1 = (objDetection[j][i].bbox.x + objDetection[j][i].bbox.width) * net_w;
            float y1 = (objDetection[j][i].bbox.y + objDetection[j][i].bbox.height) * net_w;
            x0 = (x0 - dx) / scale;
            y0 = (y0 - dy) / scale;
            x1 = (x1 - dx) / scale;
            y1 = (y1 - dy) / scale;
            x0 = (x0 > 0) ? x0 : 0;
            y0 = (y0 > 0) ? y0 : 0;
            x1 = (x1 < image_width) ? x1 : image_width - 1;
            y1 = (y1 < image_height) ? y1 : image_height - 1;
            objDetection[j][i].bbox.x = x0;
            objDetection[j][i].bbox.y = y0;
            objDetection[j][i].bbox.width = x1 - x0;
            objDetection[j][i].bbox.height = y1 - y0;
            obj.object_class = objDetection[j][i].label;
            obj.prob = objDetection[j][i].score;
            obj.bounding_box.x = objDetection[j][i].bbox.x;
            obj.bounding_box.y = objDetection[j][i].bbox.y;
            obj.bounding_box.width = objDetection[j][i].bbox.width;
            obj.bounding_box.height = objDetection[j][i].bbox.height;
            objs.push_back(obj);
        }
        arrDetection.push_back(objs);
    }
}

ClassificationRunner::ClassificationRunner(const std::string &model_path, const std::string &func_name,
                                           const std::string &names, const int device_id) {
    std::cout << "ClassificationRunner" << std::endl;
    env_.SetDeviceId(device_id);
    env_.BindDevice();
    // load offline model
    model_ = std::make_shared<edk::ModelLoader>(model_path.c_str(), func_name.c_str());
    // prepare mlu memory operator and memory
    mem_op_.SetModel(model_);
    // init easy_infer
    infer_.Init(model_, device_id);
    // create mlu resize and convert operator
    auto &in_shape = model_->InputShape(0);
    auto &out_shape = model_->OutputShape(0);
    net_w = in_shape.W();
    net_h = in_shape.H();
    net_n = in_shape.N();
    net_c = in_shape.C();
    out_n = out_shape.N();
    out_c = out_shape.C();
    out_w = out_shape.W();
    out_h = out_shape.H();
    std::cout << "net_n:" << net_n << " net_c:" << net_c << " net_h:" << net_h << " net_w:" << net_w << std::endl;
    std::cout << "out_n:" << out_n << " out_c:" << out_c << " out_h:" << out_h << " out_w:" << out_w << std::endl;

    edk::MluResizeConvertOp::Attr rc_attr;
    rc_attr.dst_h = in_shape.H();
    rc_attr.dst_w = in_shape.W();
    rc_attr.batch_size = net_n;
    rc_attr.core_version = env_.GetCoreVersion();
    rc_op_.SetMluQueue(infer_.GetMluQueue());
    if (!rc_op_.Init(rc_attr)) {
        THROW_EXCEPTION(edk::Exception::INTERNAL, rc_op_.GetLastError());
    }

    //   names
    std::ifstream fin(names, std::ios::in);
    char line[1024] = {0};
    std::string name = "";
    while (fin.getline(line, sizeof(line))) {
        std::stringstream word(line);
        word >> name;
        // std::cout << "name: " << name << std::endl;
        labels.push_back(name);
    }
    fin.clear();
    fin.close();

    // init postproc
    postproc_.reset(new edk::ClassificationPostproc);
    postproc_->set_threshold(0.2);
    CHECK(postproc_);

    mlu_output_ = mem_op_.AllocMluOutput();
    cpu_output_ = mem_op_.AllocCpuOutput();
    mluData_infer = reinterpret_cast<void **>(malloc(sizeof(void *) * 1));
    cnrtMalloc(&(mluData_infer[0]), net_n * net_c * net_w * net_h);
    //cnrtMalloc(mluData_infer, net_n* net_c* net_w* net_h);
    inputCpuPtrS = (void **) malloc(sizeof(void *) * 1);
    inputCpuPtrS[0] = (void *) malloc(net_w * net_h * net_c * net_n);
}

ClassificationRunner::~ClassificationRunner() {
    // Stop();
    if (nullptr != mlu_output_)
        mem_op_.FreeMluOutput(mlu_output_);
    if (nullptr != cpu_output_)
        mem_op_.FreeCpuOutput(cpu_output_);
    if (nullptr != mluData_infer)
        mem_op_.FreeMluInput(mluData_infer);
    if (nullptr != inputCpuPtrS)
        mem_op_.FreeCpuOutput(inputCpuPtrS);
}

int ClassificationRunner::GetBatch() { return net_n; }

void ClassificationRunner::Pre(const std::vector<cv::Mat> &vBatchImages) {
    unsigned char *ptr = (unsigned char *) inputCpuPtrS[0];
    cv::Mat alpha(net_w, net_h, CV_8UC1, cv::Scalar(255));
    // for (int j = 0; j < net_c-1; ++j) {
    //   cv::Mat channel(net_h, net_w, CV_8UC1);
    //   tmp_a.push_back(channel);
    // }
    for (int i = 0; i < net_n; i++) {
        cv::Mat img = vBatchImages[i].clone();
        cv::Mat input_image_resized;
        cv::resize(img, input_image_resized, cv::Size(net_w, net_h), 0, 0, cv::INTER_CUBIC);
        cv::Mat net_input_data_rgba(net_w, net_h, CV_8UC4, ptr);
        std::vector<cv::Mat> tmp_a;
        cv::split(input_image_resized, tmp_a);
        tmp_a.push_back(alpha);
        cv::merge(tmp_a, net_input_data_rgba);
        ptr += (net_w * net_h * net_c);
    }

    cnrtMemcpy(mluData_infer[0], inputCpuPtrS[0], net_w * net_w * net_c * net_n, CNRT_MEM_TRANS_DIR_HOST2DEV);
}

std::vector<std::vector<Prediction>> ClassificationRunner::Classify(std::vector<cv::Mat> &vBatchImages, int N) {
    // preprocess
    Pre(vBatchImages);

    // run inference
    infer_.Run(mluData_infer, mlu_output_);
    mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_);

    // post process
    std::vector<std::vector<edk::DetectObject>> detect_result;
    std::vector<std::pair<float *, uint64_t>> postproc_param;
    postproc_param.push_back(
            std::make_pair(reinterpret_cast<float *>(cpu_output_[0]), model_->OutputShape(0).DataCount()));
    detect_result = postproc_->Execute(postproc_param, net_n);

    std::vector<std::vector<Prediction>> result;
    for (int j = 0; j < net_n; ++j) {
        std::vector<Prediction> p;
        int len_result = detect_result[j].size();
        edk::DetectObject max_result = detect_result[j][0];
        for (int i = 0; i < len_result; i++) {
            if (max_result.score < detect_result[j][i].score) {
                max_result = detect_result[j][i];
            }
        }
        p.push_back(std::make_pair(labels[max_result.label], prob_sigmoid(max_result.score)));
        result.push_back(p);
    }
    return result;
}

CrowdCountPredictor::CrowdCountPredictor(const std::string &model_path, const std::string &func_name,
                                         const int device_id) {
    // set mlu environment
    std::cout << "CrowdCountPredictor" << std::endl;
    env_.SetDeviceId(device_id);
    env_.BindDevice();

    // load offline model
    model_ = std::make_shared<edk::ModelLoader>(model_path.c_str(), func_name.c_str());
    // prepare mlu memory operator and memory
    mem_op_.SetModel(model_);
    // init easy_infer
    infer_.Init(model_, device_id);
    // create mlu resize and convert operator
    auto &in_shape = model_->InputShape(0);
    auto &out_shape = model_->OutputShape(0);
    net_w = in_shape.W();
    net_h = in_shape.H();
    net_n = in_shape.N();
    net_c = in_shape.C();
    out_n = out_shape.N();
    out_c = out_shape.C();
    out_w = out_shape.W();
    out_h = out_shape.H();
    std::cout << "net_n:" << net_n << " net_c:" << net_c << " net_h:" << net_h << " net_w:" << net_w << std::endl;
    std::cout << "out_n:" << out_n << " out_c:" << out_c << " out_h:" << out_h << " out_w:" << out_w << std::endl;

    mlu_input_ = mem_op_.AllocMluInput();
    cpuData_ = mem_op_.AllocCpuInput();
    mlu_output_ = mem_op_.AllocMluOutput();
    cpu_output_ = mem_op_.AllocCpuOutput();
}

CrowdCountPredictor::~CrowdCountPredictor() {
    // Stop();
    if (nullptr != mlu_output_)
        mem_op_.FreeMluOutput(mlu_output_);
    if (nullptr != cpuData_)
        mem_op_.FreeCpuOutput(cpuData_);
    if (nullptr != cpu_output_)
        mem_op_.FreeCpuOutput(cpu_output_);
    if (nullptr != mlu_input_)
        mem_op_.FreeMluInput(mlu_input_);
}

int CrowdCountPredictor::GetBatch() { return net_n; }

void CrowdCountPredictor::Pre(cv::Mat &img) {
    // std::cout<<"Pre"<<std::endl;
    cv::Mat img_clone = img.clone();
    cv::resize(img_clone, img_clone, cv::Size(net_w, net_h));
    cvtColor(img_clone, img_clone, cv::COLOR_BGR2RGB);
    img_clone.convertTo(img_clone, CV_32FC3, 1 / 255.0);

    cv::Mat final_mean(img_clone.size(), CV_32FC3, cv::Scalar(0.452016860247, 0.447249650955, 0.431981861591));
    cv::Mat final_var(img_clone.size(), CV_32FC3, cv::Scalar(0.23242045939, 0.224925786257, 0.221840232611));

    cv::Mat normalized_mean;
    cv::subtract(img_clone, final_mean, normalized_mean);
    cv::Mat normalized_std;
    cv::divide(normalized_mean, final_var, normalized_std);
    cv::Mat norm_image;
    normalized_std.convertTo(norm_image, CV_32FC3);

    float *data = reinterpret_cast<float *>(cpuData_[0]);
    const float *indata = normalized_std.ptr<float>(0);
    for (int i = 0; i < net_n * net_c * net_w * net_h; i++) {
        data[i] = (float) *indata++;
    }
}

void CrowdCountPredictor::run(cv::Mat &images, cv::Mat &result) {
    Pre(images);

    // float* data2 = reinterpret_cast<float*>(cpuData_[0]);
    // for (int i=0;i<10;i++)
    // {
    //   std::cout<<"endinp[i]:"<<(float)data2[i]<<std::endl;
    // }

    cnrtMemcpy(mlu_input_[0], cpuData_[0], net_n * net_c * net_w * net_h * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV);

    // run inference
    infer_.Run(mlu_input_, mlu_output_);
    mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_);
    float *out = reinterpret_cast<float *>(cpu_output_[0]);
    cv::Mat ma = cv::Mat(out_h, out_w, CV_32FC1);
    for (int i = 0; i < (out_n * out_c * out_w * out_h); i++) {
        // if(i<10)
        // {
        //   std::cout<<"out[i]:"<<(float)out[i]<<std::endl;
        // }
        // ma.at<float>(i / out_w, i %  out_w) = out[i]*255;
        ma.at<float>(i / out_w, i % out_w) = out[i];
    }
    // size_t scale_x = input_w_ / out_w;
    // size_t scale_y = input_h_ / out_h;
    result = ma.clone();
}

ResnetSegment::ResnetSegment(const std::string &model_path, const std::string &func_name, const int device_id) {
    std::cout << "Cambricion Resnet18 FPN segment inference" << std::endl;
    env_.SetDeviceId(device_id);
    env_.BindDevice();
    std::cout << "model path : " << model_path << std::endl;
    model_ = std::make_shared<edk::ModelLoader>(model_path.c_str(), func_name.c_str());
    mem_op_.SetModel(model_);
    // init easy_infer
    infer_.Init(model_, device_id);
    auto &in_shape = model_->InputShape(0);
    auto &out_shape = model_->OutputShape(0);
    net_w = in_shape.W();
    net_h = in_shape.H();
    net_n = in_shape.N();
    net_c = in_shape.C();
    out_n = out_shape.N();
    out_c = out_shape.C();
    out_w = out_shape.W();
    out_h = out_shape.H();
    std::cout << "net_n:" << net_n << " net_c:" << net_c << " net_h:" << net_h << " net_w:" << net_w << std::endl;
    std::cout << "out_n:" << out_n << " out_c:" << out_c << " out_h:" << out_h << " out_w:" << out_w << std::endl;
    assert(out_n != 1);
    mlu_output_ = mem_op_.AllocMluInput();
    cpu_output_ = mem_op_.AllocCpuOutput();

    mluData_infer = reinterpret_cast<void **>(malloc(sizeof(void *) * 1));
    cnrtMalloc(&(mluData_infer[0]), net_n * net_c * net_w * net_h);

    inputCpuPtrS = (void **) malloc(sizeof(void *) * 1);
    inputCpuPtrS[0] = (void *) malloc(net_w * net_h * net_c * net_n);
}

ResnetSegment::~ResnetSegment() {
    if (nullptr != mlu_output_)
        mem_op_.FreeMluOutput(mlu_output_);
    if (nullptr != cpu_output_)
        mem_op_.FreeCpuOutput(cpu_output_);
    if (nullptr != mluData_infer)
        mem_op_.FreeMluInput(mluData_infer);
    if (nullptr != inputCpuPtrS)
        mem_op_.FreeCpuOutput(inputCpuPtrS);
}

void ResnetSegment::Pre(const std::vector<cv::Mat> &vBatchImages) {
    unsigned char *ptr = (unsigned char *) inputCpuPtrS[0];
    cv::Mat alpha(net_w, net_h, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < net_n; i++) {
        cv::Mat img = vBatchImages[i].clone();
        cv::Mat input_image_resized;
        cv::resize(img, input_image_resized, cv::Size(net_w, net_h), 0, 0, cv::INTER_CUBIC);
        cv::Mat net_input_data_rgba(net_w, net_h, CV_8UC4, ptr);
        std::vector<cv::Mat> tmp_a;
        cv::split(input_image_resized, tmp_a);
        std::reverse(tmp_a.begin(), tmp_a.end());
        tmp_a.push_back(alpha);
        cv::merge(tmp_a, net_input_data_rgba);
        ptr += (net_w * net_h * net_c);
    }
    cnrtMemcpy(mluData_infer[0], inputCpuPtrS[0], net_w * net_w * net_c * net_n, CNRT_MEM_TRANS_DIR_HOST2DEV);
}

void ResnetSegment::getfeat(std::vector<cv::Mat> &vBatchImages, cv::Mat &feat) {
    Pre(vBatchImages);

    infer_.Run(mluData_infer, mlu_output_);
    mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_);

    float *out = reinterpret_cast<float *>(cpu_output_[0]);

    // std::cout << "batch: " << std::endl;
    int batchSize = out_c * out_w * out_h;
    cv::Mat img = cv::Mat(out_h, out_w, CV_32FC1, cv::Scalar::all(0));
    for (int i = 0; i < out_n; i++) {
        int start = i * batchSize;
        for (int j = 0; j < (out_h * out_w); j++) {
            int row = j / out_w;
            int col = j % out_w;
            // if()
            img.at<float>(row, col) = out[start + 2 * j];
        }
    }
    img.convertTo(feat, CV_8UC1, 255.0);
    cv::cvtColor(feat, feat, cv::COLOR_GRAY2BGR);
}

void ResnetSegment::processFeat(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size &srcImageSize,
                                float smoke_thres, std::vector<std::vector<cv::Point>> &contours, bool &isSmoke) {
    int binary_pixels = 0;
    std::vector<std::vector<cv::Point>> contours_all;
    processFeat_test(feat, pts, srcImageSize, smoke_thres, contours, isSmoke, binary_pixels, contours_all);
}

void ResnetSegment::processFeat_test(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size &srcImageSize,
                                     float smoke_thres, std::vector<std::vector<cv::Point>> &contours, bool &isSmoke,
                                     int &binary_pixels, std::vector<std::vector<cv::Point>> &contours_all) {
    isSmoke = false;
    if (feat.empty())
        return;
    cv::Mat img;
    feat.convertTo(img, CV_8UC3);

    cv::Point *root_points = new cv::Point[pts.size()];
    for (size_t i = 0; i < pts.size(); i++) {
        root_points[i].x = pts[i].x * 256 / (float) srcImageSize.width;
        root_points[i].y = pts[i].y * 256 / (float) srcImageSize.height;
    }

    const cv::Point *ppt[1] = {root_points};
    int npt[] = {int(pts.size())};

    cv::Mat mask_ann, dst;
    img.copyTo(mask_ann);
    mask_ann.setTo(cv::Scalar::all(0));

    cv::fillPoly(mask_ann, ppt, npt, 1, cv::Scalar(255, 255, 255));
    // LOG(INFO)<<img.size()<<" "<<img.type()<<" "<<mask_ann.type();
    img.copyTo(dst, mask_ann);

    // imwrite("feat_roi.jpg" , dst);
    cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
    cv::threshold(dst, dst, 133, 255, cv::THRESH_BINARY);
    cv::Mat showimg = dst.clone();

    // all contours
    std::vector<cv::Vec4i> hierarchy; //
    std::vector<std::vector<cv::Point>> contours_tmp;
    cv::findContours(showimg, contours_tmp, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    int scale_x = (float) srcImageSize.width / (float) showimg.cols;
    int scale_y = (float) srcImageSize.height / (float) showimg.rows;
    // std::vector<std::vector<cv::Point>> contours_filter ;

    for (size_t i = 0; i < contours_tmp.size(); i++) {
        // filter small contours
        double area = cv::contourArea(contours_tmp[i]);
        cv::Rect rect1 = cv::boundingRect(cv::Mat(contours_tmp[i]));
        float tmp = (float) rect1.height / (float) rect1.width;
        float hwratio = tmp > 1 ? tmp : 1 / (float) tmp;
        // LOG(INFO)<<"area:"<<area<<" "<<contours_tmp[i][0]<<" hwratio:"<<hwratio;
        if (area > 10 * 10 && hwratio < 4) {
            std::vector<cv::Point> conts;
            for (size_t j = 0; j < contours_tmp[i].size(); j++) {
                conts.push_back(contours_tmp[i][j]);
            }
            contours.push_back(conts);
        }
    }
    // LOG(INFO)<<"contours_filter SIZE:"<<contours.size();

    cv::Mat dst_filter = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::drawContours(dst_filter, contours, -1, cv::Scalar(255, 255, 255), cv::FILLED);
    // Mat dst_filter;
    // dst.copyTo(dst_filter, mask_contour);

    // imwrite("dst.jpg" ,dst);
    // imwrite("dst_filter.jpg" ,dst_filter);

    // Mat dst_norm;
    // dst_filter.convertTo(dst_norm, CV_32FC1, 1.);
    delete[] root_points;

    binary_pixels = sum(dst_filter).val[0] / 255.0;
    // LOG(INFO)<<"binary_pixels:"<<binary_pixels;
    // exit(0);
    // float smoke_percent = 100-100*(float)binary_pixels / (float)(256*256);

    if (binary_pixels > smoke_thres) {
        isSmoke = true;

        for (size_t i = 0; i < contours.size(); i++) {
            for (size_t j = 0; j < contours[i].size(); j++) {
                contours[i][j].x *= scale_x;
                contours[i][j].y *= scale_y;
            }
        }
        for (size_t i = 0; i < contours_tmp.size(); i++) {
            std::vector<cv::Point> conts_final;
            for (size_t j = 0; j < contours_tmp[i].size(); j++) {
                conts_final.push_back(
                        cv::Point(contours_tmp[i][j].x * scale_x + 3, contours_tmp[i][j].y * scale_y + 3));
            }
            contours_all.push_back(conts_final);
        }
    } else {
        isSmoke = false;
        contours.clear();
    }
}

Segment::Segment(const std::string &model_path, const std::string &func_name, const int device_id) {
    device = device_id;
    std::cout << "initial Segment !" << std::endl;
    env_.SetDeviceId(device_id);
    std::cout << "setDevice success" << std::endl;
    env_.BindDevice();
    std::cout << "BindDevice success" << std::endl;
    model_ = std::make_shared<edk::ModelLoader>(model_path.c_str(), func_name.c_str());
    std::cout << "initial model success !" << std::endl;
    // prepare mlu memory operator and memory
    mem_op_.SetModel(model_);
    // init easy_infer
    infer_.Init(model_, device_id);

    // create mlu resize and convert operator
    auto &in_shape = model_->InputShape(0);
    auto &out_shape = model_->OutputShape(0);
    int outNum = (int) model_->OutputNum();

    net_w = in_shape.W();
    net_h = in_shape.H();
    net_n = in_shape.N();
    net_c = in_shape.C();

    out_n = out_shape.N();
    out_c = out_shape.C();
    out_w = out_shape.W();
    out_h = out_shape.H();

    batch_size = net_n;

    model_input_size = net_h * net_w * net_c * net_n * sizeof(uint8_t);
    // model_output_size = out_h * out_w * out_c * out_n * sizeof(int32_t);
    // model_output_size = out_h * out_w * out_c * out_n * sizeof(uint8_t);

    std::cout << "net_n:" << net_n << " net_c:" << net_c << " net_h:" << net_h << " net_w:" << net_w << std::endl;
    std::cout << "out_n:" << out_n << " out_c:" << out_c << " out_h:" << out_h << " out_w:" << out_w << std::endl;
    std::cout << "out numbers: " << outNum << std::endl;

    std::cout << "batch size: " << batch_size << std::endl;

    CNRT_SAFECALL(cnrtMalloc(&(model_input[0]), model_input_size));
    // CNRT_SAFECALL(cnrtMalloc(&(model_output[0]), model_output_size));
    // model_output.AllocMluOutput()
    model_output = mem_op_.AllocMluOutput();

    CNRT_SAFECALL(cnrtCreateQueue(&queue));
    CNCV_SAFECALL(cncvCreate(&handle));
    CNCV_SAFECALL(cncvSetQueue(handle, queue));

    src_rois = new cncvRect[batch_size];
    tmp_rois = new cncvRect[batch_size];

    CNRT_SAFECALL(cnrtMalloc((void **) &mlu_input, batch_size * sizeof(void *)));
    CNRT_SAFECALL(cnrtMalloc((void **) &tmp, batch_size * sizeof(void *)));
    CNRT_SAFECALL(cnrtMalloc((void **) &mlu_output, batch_size * sizeof(void *)));

    cpu_src_imgs = new void *[batch_size];
    cpu_src_imgs_buff_size = 100 * sizeof(uint8_t);
    std::cout << "cpu_src_imgs_buff_size: " << cpu_src_imgs_buff_size << std::endl;
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
    }

    cpu_tmp_imgs = new void *[batch_size];

    dst_size = net_w * net_h * 4 * sizeof(uint8_t);
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        CNRT_SAFECALL(cnrtMalloc(&(cpu_tmp_imgs[idx]), dst_size));
    }

    cpu_dst_imgs = new void *[batch_size];

    cpu_output_ = mem_op_.AllocCpuOutput();

    std::cout << "initialize Segment success" << std::endl;
}

Segment::~Segment() {
    if (nullptr != model_output)
        mem_op_.FreeMluOutput(model_output);
    cncvDestroy(handle);
    cnrtDestroyQueue(queue);
    for (uint32_t i = 0; i < batch_size; ++i) {
        cnrtFree(cpu_src_imgs[i]);
        cnrtFree(cpu_tmp_imgs[i]);
    }
    cnrtFree(mlu_input);
    cnrtFree(tmp);
    cnrtFree(mlu_output);
    // cnrtFree(workspace);
    delete[] cpu_src_imgs;
    delete[] cpu_tmp_imgs;
    delete[] cpu_dst_imgs;
    delete[] src_rois;
    delete[] tmp_rois;
}

int Segment::GetBatch() { return net_n; }

void Segment::getfeat(std::vector<cv::Mat> &vBatchImgs, cv::Mat &feat) {
    std::cout << "Segment inference" << std::endl;
    env_.BindDevice();
    Preprocess(vBatchImgs, net_w, net_h, model_input[0]);
    // testInput(model_input[0]);
    //检测输入
    // void** cpu_input;
    // cpu_input = mem_op_.AllocCpuInput();
    // mem_op_.MemcpyD2H(cpu_input[0], model_input[0], model_input_size);
    // std::cout << "uint8_t: " << sizeof(uint8_t) << std::endl;

    // cv::Mat img = cv::Mat(256 * 4, 256, CV_8UC4, cv::Scalar::all(0));

    // memcpy(img.data, (reinterpret_cast<uint8_t*>(cpu_input[0])), 256 * 256 * 4 * 4);
    // cv::Mat imgout;
    // cv::cvtColor(img, imgout, cv::COLOR_BGRA2BGR);
    // cv::imwrite("/data/ld/project/git/samples/test/ttt.png", imgout);

    // std::cout << "prePare over" << std::endl;
    infer_.Run(model_input, model_output);
    // std::cout << "Run over" << std::endl;
    mem_op_.MemcpyOutputD2H(cpu_output_, model_output);
    // std::cout << "mem_op_ over" << std::endl;
    float *out = reinterpret_cast<float *>(cpu_output_[0]);
    // std::cout << "batch: " << std::endl;
    int batchSize = out_c * out_w * out_h;
    for (int i = 0; i < out_n; i++) {
        cv::Mat img = cv::Mat(out_h, out_w, CV_32FC1, cv::Scalar::all(0));
        int start = i * batchSize;
        for (int j = 0; j < (out_h * out_w); j++) {
            int row = j / out_w;
            int col = j % out_w;
            // if()
            img.at<float>(row, col) = out[start + 2 * j];
        }
        img.convertTo(feat, CV_8UC1, 255.0);
        cv::cvtColor(feat, feat, cv::COLOR_GRAY2BGR);
    }
}

void Segment::Preprocess(const std::vector<cv::Mat> &imgs, int dst_w, int dst_h, void *output) {
    if (imgs.empty()) {
        std::cerr << "no image" << std::endl;
        abort();
    }
    // std::cout << "start preprocessing" << std::endl;
    assert(batch_size == imgs.size());
    uint32_t maxSize = 0;
    for (int i = 0; i < imgs.size(); ++i) {
        uint32_t tmpSize = imgs[i].rows * imgs[i].step * sizeof(uint8_t);
        if (tmpSize > maxSize) {
            maxSize = tmpSize;
        }
    }
    // std::cout << "max size: " << maxSize << std::endl;
    cncvRect dst_roi;

    void *workspace;
    size_t workspace_size;

    int src_w = imgs[0].cols;
    int src_h = imgs[0].rows;
    int src_stride = imgs[0].step;
    // std::cout << " src_w: " << src_w << " src_h: " << src_h << " src_stride: " << src_stride << std::endl;

    while (maxSize > cpu_src_imgs_buff_size) {
        for (uint32_t i = 0; i < batch_size; ++i) {
            cnrtFree(cpu_src_imgs[i]);
            std::cout << "free histroy mlu memry" << std::endl;
        }
        cpu_src_imgs_buff_size = maxSize + 128;
        for (uint32_t idx = 0; idx < batch_size; ++idx) {
            CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
            std::cout << "remalloc mlu memory" << std::endl;
        }
    }
    uint32_t src_size;
    // copy src imgs to mlu
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        // CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), src_size));
        src_size = src_h * src_stride * sizeof(uint8_t);
        CNRT_SAFECALL(cnrtMemcpy(cpu_src_imgs[idx], imgs[idx].data, src_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    }

    CNRT_SAFECALL(cnrtMemcpy(mlu_input, cpu_src_imgs, batch_size * sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV));

    CNRT_SAFECALL(cnrtMemcpy(tmp, cpu_tmp_imgs, batch_size * sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV));
    for (uint32_t idx = 0; idx < batch_size; ++idx) {
        cpu_dst_imgs[idx] = reinterpret_cast<uint8_t *>(output) + idx * dst_size;
    }
    CNRT_SAFECALL(cnrtMemcpy(mlu_output, cpu_dst_imgs, batch_size * sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV));

    CNCV_SAFECALL(cncvGetResizeRgbxWorkspaceSize(batch_size, &workspace_size));
    CNRT_SAFECALL(cnrtMalloc(&workspace, workspace_size));

    src_desc.width = src_w;
    src_desc.height = src_h;
    src_desc.pixel_fmt = CNCV_PIX_FMT_BGR;
    src_desc.stride[0] = src_stride;
    src_desc.depth = CNCV_DEPTH_8U;

    tmp_desc.width = dst_w;
    tmp_desc.height = dst_h;
    tmp_desc.pixel_fmt = CNCV_PIX_FMT_BGR;
    tmp_desc.stride[0] = dst_w * 3 * sizeof(uint8_t);
    tmp_desc.depth = CNCV_DEPTH_8U;

    for (uint32_t i = 0; i < batch_size; ++i) {
        // init dst rect
        tmp_rois[i].x = 0;
        tmp_rois[i].y = 0;
        tmp_rois[i].w = dst_w;
        tmp_rois[i].h = dst_h;

        // init src rect
        src_rois[i].x = 0;
        src_rois[i].y = 0;
        src_rois[i].w = src_w;
        src_rois[i].h = src_h;
        // if (keep_aspect_ratio) {
        //   KeepAspectRatio(&tmp_rois[i], src_desc, tmp_desc);
        // }
    }
    CNCV_SAFECALL(cncvResizeRgbx(handle, batch_size, src_desc, src_rois, mlu_input, tmp_desc, tmp_rois, tmp,
                                 workspace_size, workspace, CNCV_INTER_BILINEAR));
    dst_roi = tmp_rois[0];
    dst_desc = tmp_desc;
    dst_desc.pixel_fmt = CNCV_PIX_FMT_BGRA;
    dst_desc.stride[0] = dst_w * 4 * sizeof(uint8_t);
    // std::cout << "async" << std::endl;
    CNCV_SAFECALL(cncvRgbxToRgbx(handle, batch_size, tmp_desc, dst_roi, tmp, dst_desc, dst_roi, mlu_output));

    // wait for task finished
    CNRT_SAFECALL(cnrtSyncQueue(queue));
    cnrtFree(workspace);
}

void Segment::processFeat(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size &srcImageSize,
                          float smoke_thres, std::vector<std::vector<cv::Point>> &contours, bool &isSmoke) {
    int binary_pixels = 0;
    std::vector<std::vector<cv::Point>> contours_all;
    processFeat_test(feat, pts, srcImageSize, smoke_thres, contours, isSmoke, binary_pixels, contours_all);
}

void Segment::processFeat_test(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size &srcImageSize,
                               float smoke_thres, std::vector<std::vector<cv::Point>> &contours, bool &isSmoke,
                               int &binary_pixels, std::vector<std::vector<cv::Point>> &contours_all) {
    isSmoke = false;
    if (feat.empty())
        return;
    cv::Mat img;
    feat.convertTo(img, CV_8UC3);

    cv::Point *root_points = new cv::Point[pts.size()];
    for (size_t i = 0; i < pts.size(); i++) {
        root_points[i].x = pts[i].x * 256 / (float) srcImageSize.width;
        root_points[i].y = pts[i].y * 256 / (float) srcImageSize.height;
    }

    const cv::Point *ppt[1] = {root_points};
    int npt[] = {int(pts.size())};

    cv::Mat mask_ann, dst;
    img.copyTo(mask_ann);
    mask_ann.setTo(cv::Scalar::all(0));

    cv::fillPoly(mask_ann, ppt, npt, 1, cv::Scalar(255, 255, 255));
    // LOG(INFO)<<img.size()<<" "<<img.type()<<" "<<mask_ann.type();
    img.copyTo(dst, mask_ann);

    // imwrite("feat_roi.jpg" , dst);
    cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
    cv::threshold(dst, dst, 133, 255, cv::THRESH_BINARY);
    cv::Mat showimg = dst.clone();

    // all contours
    std::vector<cv::Vec4i> hierarchy; //
    std::vector<std::vector<cv::Point>> contours_tmp;
    cv::findContours(showimg, contours_tmp, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    int scale_x = (float) srcImageSize.width / (float) showimg.cols;
    int scale_y = (float) srcImageSize.height / (float) showimg.rows;
    // std::vector<std::vector<cv::Point>> contours_filter ;

    for (size_t i = 0; i < contours_tmp.size(); i++) {
        // filter small contours
        double area = cv::contourArea(contours_tmp[i]);
        cv::Rect rect1 = cv::boundingRect(cv::Mat(contours_tmp[i]));
        float tmp = (float) rect1.height / (float) rect1.width;
        float hwratio = tmp > 1 ? tmp : 1 / (float) tmp;
        // LOG(INFO)<<"area:"<<area<<" "<<contours_tmp[i][0]<<" hwratio:"<<hwratio;
        if (area > 10 * 10 && hwratio < 4) {
            std::vector<cv::Point> conts;
            for (size_t j = 0; j < contours_tmp[i].size(); j++) {
                conts.push_back(contours_tmp[i][j]);
            }
            contours.push_back(conts);
        }
    }
    // LOG(INFO)<<"contours_filter SIZE:"<<contours.size();

    cv::Mat dst_filter = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::drawContours(dst_filter, contours, -1, cv::Scalar(255, 255, 255), cv::FILLED);
    // Mat dst_filter;
    // dst.copyTo(dst_filter, mask_contour);

    // imwrite("dst.jpg" ,dst);
    // imwrite("dst_filter.jpg" ,dst_filter);

    // Mat dst_norm;
    // dst_filter.convertTo(dst_norm, CV_32FC1, 1.);
    delete[] root_points;

    binary_pixels = sum(dst_filter).val[0] / 255.0;
    // LOG(INFO)<<"binary_pixels:"<<binary_pixels;
    // exit(0);
    // float smoke_percent = 100-100*(float)binary_pixels / (float)(256*256);

    if (binary_pixels > smoke_thres) {
        isSmoke = true;

        for (size_t i = 0; i < contours.size(); i++) {
            for (size_t j = 0; j < contours[i].size(); j++) {
                contours[i][j].x *= scale_x;
                contours[i][j].y *= scale_y;
            }
        }
        for (size_t i = 0; i < contours_tmp.size(); i++) {
            std::vector<cv::Point> conts_final;
            for (size_t j = 0; j < contours_tmp[i].size(); j++) {
                conts_final.push_back(
                        cv::Point(contours_tmp[i][j].x * scale_x + 3, contours_tmp[i][j].y * scale_y + 3));
            }
            contours_all.push_back(conts_final);
        }
    } else {
        isSmoke = false;
        contours.clear();
    }
}

// int OpencvMtcnn::load_model(const std::string& modelfolder, const std::string &func_name,int ngpuid) 
// {
//     std::vector<std::string> pnet_model_path{modelfolder + "/det1_576x324_int8.cambricon",
//                                              modelfolder + "/det1_409x230_int8.cambricon",
//                                              modelfolder + "/det1_290x163_int8.cambricon",
//                                              modelfolder + "/det1_206x116_int8.cambricon",
//                                              modelfolder + "/det1_146x82_int8.cambricon",
//                                              modelfolder + "/det1_104x59_int8.cambricon",
//                                              modelfolder + "/det1_74x42_int8.cambricon",
//                                              modelfolder + "/det1_52x30_int8.cambricon",
//                                              modelfolder + "/det1_37x21_int8.cambricon",
//                                              modelfolder + "/det1_27x15_int8.cambricon"};
//     std::string rnet_model_path = modelfolder + "/det2_16batch_int8.cambricon";
//     std::string onet_model_path = modelfolder + "/det3_16batch_int8.cambricon";                                        
//     for (int i = 0; i < pnet_model_num; i++) {
    
//         pnet_env[i].SetDeviceId(ngpuid);
//         pnet_env[i].BindDevice();
//         pnet_model[i] = std::make_shared<edk::ModelLoader>(pnet_model_path[i].c_str(), func_name.c_str());
//         pnet_model_mem_op[i].SetModel(pnet_model[i]);
//         pnet_model_infer[i].Init(pnet_model[i], ngpuid);
//         auto &in_shape = pnet_model[i]->InputShape(0);
//         auto &reg_shape = pnet_model[i]->OutputShape(0);
//         auto &confidence_shape = pnet_model[i]->OutputShape(1);
//         pnet_w[i] = in_shape.W();
//         pnet_h[i] = in_shape.H();
//         pnet_n[i] = in_shape.N();
//         pnet_c[i] = in_shape.C();
//         reg_n[i] = reg_shape.N();
//         reg_c[i] = reg_shape.C();
//         reg_h[i] = reg_shape.H();
//         reg_w[i] = reg_shape.W();
   
//         edk::MluResizeConvertOp::Attr rc_attr;
//         rc_attr.dst_h = in_shape.H();
//         rc_attr.dst_w = in_shape.W();
//         rc_attr.batch_size = pnet_n[i];
//         rc_attr.core_version = pnet_env[i].GetCoreVersion();
//         pnet_rc_op[i].SetMluQueue(pnet_model_infer[i].GetMluQueue());
//         if (!pnet_rc_op[i].Init(rc_attr)) {
//             THROW_EXCEPTION(edk::Exception::INTERNAL, pnet_rc_op[i].GetLastError());
//         }
 
//         pnet_cpu_input[i] = pnet_model_mem_op[i].AllocCpuInput();
//         pnet_mlu_input[i] = pnet_model_mem_op[i].AllocMluInput();
//         pnet_mlu_output[i] = pnet_model_mem_op[i].AllocMluOutput();
//         pnet_cpu_output[i] = pnet_model_mem_op[i].AllocCpuOutput();
        
//         reg_cout_trans[i] = (float*)malloc(reg_n[i] * reg_w[i] * reg_h[i]  * 4 * sizeof(float));
//         confidence_trans[i] = (float*)malloc(reg_n[i] * reg_w[i] * reg_h[i] * 2 * sizeof(float));
        
//     }

//     rnet_env.SetDeviceId(ngpuid);
//     rnet_env.BindDevice();
//     rnet_model = std::make_shared<edk::ModelLoader>(rnet_model_path.c_str(), func_name.c_str());
//     rnet_model_mem_op.SetModel(rnet_model);
//     rnet_model_infer.Init(rnet_model, ngpuid);

//     auto &in_shape = rnet_model->InputShape(0);
//     auto &reg_shape = rnet_model->OutputShape(0);
//     auto &confidence_shape = rnet_model->OutputShape(1);
    
//     rnet_w = in_shape.W();
//     rnet_h = in_shape.H();
//     rnet_n = in_shape.N();
//     rnet_c = in_shape.C();
//     rnet_reg_n = reg_shape.N();
//     rnet_reg_c = reg_shape.C();
//     rnet_reg_w = reg_shape.W();
//     rnet_reg_h = reg_shape.H();

//     edk::MluResizeConvertOp::Attr rc_attr;
//     rc_attr.dst_h = in_shape.H();
//     rc_attr.dst_w = in_shape.W();
//     rc_attr.batch_size = rnet_n;
//     rc_attr.core_version = rnet_env.GetCoreVersion();
//     rnet_rc_op.SetMluQueue(rnet_model_infer.GetMluQueue());
//     if (!rnet_rc_op.Init(rc_attr)) {
//         THROW_EXCEPTION(edk::Exception::INTERNAL, rnet_rc_op.GetLastError());
//     }
//     rnet_cpu_input = rnet_model_mem_op.AllocCpuInput();
//     rnet_cpu_output = rnet_model_mem_op.AllocCpuOutput();
//     rnet_mlu_input = rnet_model_mem_op.AllocMluInput();
//     rnet_mlu_output = rnet_model_mem_op.AllocMluOutput();
//     rnet_reg_cout_trans = (float*)malloc(rnet_reg_n*4*rnet_reg_w*rnet_reg_h*sizeof(float));
//     rnet_confidence_trans = (float*)malloc(rnet_reg_n*2*rnet_reg_w*rnet_reg_h*sizeof(float));
//     rnet_input_trans = malloc(rnet_n*rnet_c*rnet_h*rnet_w);


//     onet_env.SetDeviceId(ngpuid);
//     onet_env.BindDevice();
//     onet_model = std::make_shared<edk::ModelLoader>(onet_model_path.c_str(), func_name.c_str());
//     onet_model_mem_op.SetModel(onet_model);
//     onet_model_infer.Init(onet_model, ngpuid);
    
//     auto &onet_in_shape = onet_model->InputShape(0);
//     auto &onet_reg_shape = onet_model->OutputShape(0);
//     auto &onet_landmark_shape = onet_model->OutputShape(1);
//     auto &onet_confidence_shape = onet_model->OutputShape(2);

//     onet_w = onet_in_shape.W();
//     onet_h = onet_in_shape.H();
//     onet_n = onet_in_shape.N();
//     onet_c = onet_in_shape.C();
//     onet_reg_n = onet_reg_shape.N();
//     onet_reg_c = onet_reg_shape.C();
//     onet_reg_w = onet_reg_shape.W();
//     onet_reg_h = onet_reg_shape.H();

//     edk::MluResizeConvertOp::Attr onet_attr;
//     onet_attr.dst_h = onet_h;
//     onet_attr.dst_w = onet_w;
//     onet_attr.batch_size = onet_n;
//     onet_attr.core_version = onet_env.GetCoreVersion();
//     onet_rc_op.SetMluQueue(onet_model_infer.GetMluQueue());
//     if (!onet_rc_op.Init(onet_attr)) 
//     {
//         THROW_EXCEPTION(edk::Exception::INTERNAL, onet_rc_op.GetLastError());
//     }

//     onet_cpu_input = onet_model_mem_op.AllocCpuInput();
//     onet_cpu_output = onet_model_mem_op.AllocCpuOutput();
//     onet_mlu_input = onet_model_mem_op.AllocMluInput();
//     onet_mlu_output = onet_model_mem_op.AllocMluOutput();

//     onet_reg_cout_trans = (float*)malloc(onet_reg_n*onet_reg_c*onet_reg_w*onet_reg_h*sizeof(float));
//     onet_confidence_trans = (float*)malloc(onet_reg_n*2*onet_reg_w*onet_reg_h*sizeof(float));
//     onet_landmark_trans = (float*)malloc(onet_reg_n*10*onet_reg_w*onet_reg_h*sizeof(float));
//     onet_input_trans = malloc(onet_n*onet_c*onet_h*onet_w);
   
//     return 0;
// }



// void OpencvMtcnn::nms(std::vector<BoundingBox> *boxes, float threshold,int type, std::vector<BoundingBox> *filterOutBoxes) 
// {
//   filterOutBoxes->clear();
//   if ((*boxes).size() == 0) return;

//   // descending sort
//   sort((*boxes).begin(), (*boxes).end(), CmpBoundingBox());
//   std::vector<size_t> idx((*boxes).size());
//   for (int i = 0; i < idx.size(); i++) {
//     idx[i] = i;
//   }
//   while (idx.size() > 0) {
//     int good_idx = idx[0];
//     filterOutBoxes->push_back((*boxes)[good_idx]);
//     // hypothesis : the closer the scores are similar
//     std::vector<size_t> tmp = idx;
//     idx.clear();
//     for (int i = 1; i < tmp.size(); i++) {
//       int tmp_i = tmp[i];
//       float inter_x1 = std::max((*boxes)[good_idx].x1, (*boxes)[tmp_i].x1);
//       float inter_y1 = std::max((*boxes)[good_idx].y1, (*boxes)[tmp_i].y1);
//       float inter_x2 = std::min((*boxes)[good_idx].x2, (*boxes)[tmp_i].x2);
//       float inter_y2 = std::min((*boxes)[good_idx].y2, (*boxes)[tmp_i].y2);

//       float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
//       float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

//       float inter_area = w * h;
//       float area_1 = ((*boxes)[good_idx].x2 - (*boxes)[good_idx].x1 + 1) *
//                      ((*boxes)[good_idx].y2 - (*boxes)[good_idx].y1 + 1);
//       float area_2 =
//           ((*boxes)[i].x2 - (*boxes)[i].x1 + 1) * ((*boxes)[i].y2 - (*boxes)[i].y1 + 1);
//       float o = (type == 1 ? (inter_area / (area_1 + area_2 - inter_area))
//                                : (inter_area / std::min(area_1, area_2)));
//       if (o <= threshold) idx.push_back(tmp_i);
//     }
//   }
// }


// int OpencvMtcnn::run_PNet(const cv::Mat &img, int i, std::vector<BoundingBox>& nmsOutBoxes) 
// {
//     unsigned char *ptr = (unsigned char *) (*pnet_cpu_input[i]);
//     cv::Mat resized;
//     int scale_h = pnet_h[i];
//     int scale_w = pnet_w[i];
    
//     cv::resize(img, resized, cv::Size(scale_w, scale_h), cv::INTER_NEAREST);
//     cv::Mat alpha = cv::Mat(scale_h, scale_w, CV_8UC1, cv::Scalar(0));

//     for (int k = 0; k < pnet_n[i]; k++) {
//         std::vector<cv::Mat> input_channels;
//         cv::split(resized, input_channels);
//         input_channels.push_back(alpha);
//         cv::Mat input_img = cv::Mat(scale_h, scale_w, CV_8UC4, ptr);
//         cv::merge(input_channels, input_img);
//         ptr += scale_w * scale_h * pnet_c[i];
//     }
//     int dim_order[4] = {0, 3, 1, 2};
//     int dim_input_order[4] = {0, 1, 2, 3};
//     int dim_input_shape[4] = {pnet_n[i],pnet_h[i],pnet_w[i],4};
//     int dim_shape[4] = {reg_n[i], reg_h[i], reg_w[i], 4};
//     int dim_shape_2[4] = {reg_n[i], reg_h[i], reg_w[i], 2};
 
//     cnrtMemcpy(pnet_mlu_input[i][0], pnet_cpu_input[i][0], pnet_n[i] * pnet_w[i] * pnet_h[i] * pnet_c[i] , CNRT_MEM_TRANS_DIR_HOST2DEV);
//     pnet_model_infer[i].Run(pnet_mlu_input[i], pnet_mlu_output[i]);
//     pnet_model_mem_op[i].MemcpyOutputD2H(pnet_cpu_output[i], pnet_mlu_output[i]);
     
//     cnrtTransOrderAndCast(pnet_cpu_output[i][0],CNRT_FLOAT32,reg_cout_trans[i],CNRT_FLOAT32,NULL,4,dim_shape,dim_order);
//     cnrtTransOrderAndCast(pnet_cpu_output[i][1],CNRT_FLOAT32,confidence_trans[i],CNRT_FLOAT32,NULL,4,dim_shape_2,dim_order);
 
//     std::vector<BoundingBox> filterOutBoxes;
    
//     float cur_sc_w = 1.0 * pnet_w[i] / timg_width;
//     float cur_sc_h = 1.0 * pnet_h[i] / timg_height;
//     generateBoundingBox(reg_cout_trans[i], i, confidence_trans[i], cur_sc_w, cur_sc_h,pnet_threshold_, filterOutBoxes);
//     nms(&filterOutBoxes, 0.5, 1, &nmsOutBoxes);
// }




// void buildInputChannels(const vector<cv::Mat> &img_channels,const vector<BoundingBox> &boxes,const cv::Size &target_size,vector<cv::Mat> *input_channels) 
// {
//   assert(img_channels.size() * boxes.size() == input_channels->size());
//   cv::Rect img_rect(0, 0, img_channels[0].cols, img_channels[0].rows);
//   for (int n = 0; n < boxes.size(); n++) 
//   {
//     cv::Rect rect;
//     rect.x = boxes[n].x1;
//     rect.y = boxes[n].y1;
//     rect.width = boxes[n].x2 - boxes[n].x1 + 1;
//     rect.height = boxes[n].y2 - boxes[n].y1 + 1;
//     cv::Rect cuted_rect = rect & img_rect;
//     cv::Rect inner_rect(cuted_rect.x - rect.x, cuted_rect.y - rect.y,cuted_rect.width, cuted_rect.height);
//     for (int c = 0; c < img_channels.size(); c++) 
//     {
//       int type = CV_8UC1;
//       cv::Mat tmp =  cv::Mat(rect.height, rect.width, type, cv::Scalar(0));
//       img_channels[c](cuted_rect).copyTo((tmp)(inner_rect));
//       cv::resize(tmp, (*input_channels)[n * img_channels.size() + c],target_size);
//     }
//   }
// }


// void OpencvMtcnn::run_RNet(const cv::Mat &img, std::vector<BoundingBox> &pnet_boxes, std::vector<BoundingBox> &output_boxes)
// {
//     const int loop_num = pnet_boxes.size() / rnet_n;
//     const int rem = pnet_boxes.size() % rnet_n;
//     std::vector<cv::Mat> channels;
//     unsigned char* rnet_cpu_input_tmp = (unsigned char*)rnet_cpu_input[0];
//     for(int i = 0; i < rnet_n;i++)
//     {
//         for(int j = 0; j < rnet_c; j++)
//         {
//             cv::Mat channel(rnet_h,rnet_w,CV_8UC1,rnet_cpu_input_tmp);
//             channels.push_back(channel);
//             rnet_cpu_input_tmp+= rnet_h*rnet_w;
//         }
//     }
//     std::vector<cv::Mat> sample_norm_channels;
//     cv::split(img,sample_norm_channels);
//     cv::Mat alpha = cv::Mat(img.rows,img.cols,CV_8UC1,cv::Scalar(0));
//     sample_norm_channels.push_back(alpha);
//     std::vector<BoundingBox> batchBoxs;
//     vector<float> rnet_cls_all, rnet_regs_all;
//     for(int i = 0; i < loop_num; i++)
//     {
//         batchBoxs.clear();
//         for(int j = 0; j < rnet_n; j++)
//         {
//             batchBoxs.push_back(total_pnet_boxes.at(j + i * rnet_n));
//         }
//         buildInputChannels(sample_norm_channels, batchBoxs, cv::Size(24, 24),&channels);
//         int dim_order[4] = {0, 3, 1, 2};
//         int dim_input_order[4] = {0, 1, 2, 3};
//         int dim_input_transorder[4] = {0 , 2 , 3 , 1};
//         int dim_input_shape[4] = {rnet_n,rnet_c,rnet_h,rnet_w};
//         int dim_shape[4] = {rnet_reg_n, rnet_reg_h, rnet_reg_w, 4};
//         int dim_shape_2[4] = {rnet_reg_n, rnet_reg_h, rnet_reg_w, 2};
//         cnrtTransDataOrder(rnet_cpu_input[0],CNRT_UINT8,rnet_input_trans,4,dim_input_shape,dim_input_transorder);

//         cnrtMemcpy(rnet_mlu_input[0],rnet_input_trans,rnet_n * rnet_c * rnet_w * rnet_h,CNRT_MEM_TRANS_DIR_HOST2DEV);
//         rnet_model_infer.Run(rnet_mlu_input,rnet_mlu_output);
//         rnet_model_mem_op.MemcpyOutputD2H(rnet_cpu_output,rnet_mlu_output);

//         cnrtTransOrderAndCast(rnet_cpu_output[0],CNRT_FLOAT32,rnet_reg_cout_trans,CNRT_FLOAT32,NULL,4,dim_shape,dim_order);
//         cnrtTransOrderAndCast(rnet_cpu_output[1],CNRT_FLOAT32,rnet_confidence_trans,CNRT_FLOAT32,NULL,4,dim_shape_2,dim_order);

//         const float *begin0 = rnet_reg_cout_trans;
//         const float *end0 = rnet_reg_n * rnet_reg_h*rnet_reg_c*rnet_reg_w+ begin0;
//         vector<float> rnet_regs(begin0, end0);
//         rnet_regs_all.insert(rnet_regs_all.end(), rnet_regs.begin(),rnet_regs.end());

//         const float *begin1 = rnet_confidence_trans;
//         const float *end1 =  rnet_reg_n * rnet_reg_h*2*rnet_reg_w + begin1;
//         vector<float> rnet_cls(begin1, end1);
//         rnet_cls_all.insert(rnet_cls_all.end(), rnet_cls.begin(), rnet_cls.end());
//     }
//     if (rem > 0)
//     {
//       batchBoxs.clear();
//       for (int j = 0; j < rem; j++) 
//       {
//         batchBoxs.push_back(pnet_boxes.at(j + loop_num * rnet_n));
//       }
//       for (int j = rem; j < rnet_n; j++) 
//       {
//         batchBoxs.push_back(pnet_boxes.at(pnet_boxes.size() - 1));
//       }
//       buildInputChannels(sample_norm_channels, batchBoxs, cv::Size(24, 24),&channels);
//       int dim_order[4] = {0, 3, 1, 2};
//       int dim_input_order[4] = {0, 1, 2, 3};
//       int dim_input_transorder[4] = {0 , 2 , 3 , 1};
//       int dim_input_shape[4] = {rnet_n,rnet_c,rnet_h,rnet_w};
//       int dim_shape[4] = {rnet_reg_n, rnet_reg_h, rnet_reg_w, 4};
//       int dim_shape_2[4] = {rnet_reg_n, rnet_reg_h, rnet_reg_w, 2};
//       cnrtTransDataOrder(rnet_cpu_input[0],CNRT_UINT8,rnet_input_trans,4,dim_input_shape,dim_input_transorder);
//       cnrtMemcpy(rnet_mlu_input[0],rnet_input_trans,rnet_n * rnet_c * rnet_w * rnet_h,CNRT_MEM_TRANS_DIR_HOST2DEV);
//       rnet_model_infer.Run(rnet_mlu_input,rnet_mlu_output);
//       rnet_model_mem_op.MemcpyOutputD2H(rnet_cpu_output,rnet_mlu_output);

//       cnrtTransOrderAndCast(rnet_cpu_output[0],CNRT_FLOAT32,rnet_reg_cout_trans,CNRT_FLOAT32,NULL,4,dim_shape,dim_order);
//       cnrtTransOrderAndCast(rnet_cpu_output[1],CNRT_FLOAT32,rnet_confidence_trans,CNRT_FLOAT32,NULL,4,dim_shape_2,dim_order);

//       const float *begin0 = rnet_reg_cout_trans;
//       const float *end0 = rnet_reg_n * rnet_reg_h*rnet_reg_c*rnet_reg_w+ begin0;
//       vector<float> rnet_regs(begin0, end0);
//       rnet_regs_all.insert(rnet_regs_all.end(), rnet_regs.begin(),rnet_regs.end());

//       const float *begin1 = rnet_confidence_trans;
//       const float *end1 =  rnet_reg_n * rnet_reg_h*2*rnet_reg_w + begin1;
//       vector<float> rnet_cls(begin1, end1);
//       rnet_cls_all.insert(rnet_cls_all.end(), rnet_cls.begin(), rnet_cls.end());
//     }
//     output_boxes.clear();
//     filteroutBoundingBox(pnet_boxes, rnet_regs_all, rnet_cls_all, std::vector<float>(),std::vector<int>(), rnet_threshold_, &output_boxes);
//     nms(&output_boxes, 0.7, 1, &pnet_boxes);
//     output_boxes.clear();
//     if (pnet_boxes.size() > 0) 
//     {
//       vector<BoundingBox> globalFilterBoxes(pnet_boxes);
//       pnet_boxes.clear();
//       for (int i = 0; i < globalFilterBoxes.size(); i++) 
//       {
//         if (globalFilterBoxes.at(i).x1 > 0 && globalFilterBoxes.at(i).x1 < globalFilterBoxes.at(i).x2 - m_min_size && globalFilterBoxes.at(i).y1 > 0 && globalFilterBoxes.at(i).y1 < globalFilterBoxes.at(i).y2 - m_min_size &&
//             globalFilterBoxes.at(i).x2 < timg_width &&
//             globalFilterBoxes.at(i).y2 < timg_height) {
//           output_boxes.push_back(globalFilterBoxes.at(i));
//         }
//       }
//     }
  
// }


// void OpencvMtcnn::filteroutBoundingBox(const vector<BoundingBox> &boxes,const vector<float> &boxRegs, const vector<float> &cls, const vector<float> &points, const vector<int> &points_shape,float threshold, vector<BoundingBox> *filterOutBoxes) 
// {
//   filterOutBoxes->clear();
//   for (int i = 0; i < boxes.size(); i++) {
//     float score = cls[i * 2 + 1];
//     if (score > threshold) {
//       BoundingBox box = boxes[i];
//       float w = boxes[i].y2 - boxes[i].y1 + 1;
//       float h = boxes[i].x2 - boxes[i].x1 + 1;
//       if (points.size() > 0) {
//         for (int p = 0; p < 5; p++) {
//           box.points_x[p] = points[i * 10 + 5 + p] * w + boxes[i].x1 - 1;
//           box.points_y[p] = points[i * 10 + p] * h + boxes[i].y1 - 1;
//         }
//       }
//       box.dx1 = boxRegs[i * 4 + 0];
//       box.dy1 = boxRegs[i * 4 + 1];
//       box.dx2 = boxRegs[i * 4 + 2];
//       box.dy2 = boxRegs[i * 4 + 3];

//       box.x1 = boxes[i].x1 + box.dy1 * w;
//       box.y1 = boxes[i].y1 + box.dx1 * h;
//       box.x2 = boxes[i].x2 + box.dy2 * w;
//       box.y2 = boxes[i].y2 + box.dx2 * h;

//       // rerec
//       w = box.x2 - box.x1;
//       h = box.y2 - box.y1;
//       float l = std::max(w, h);
//       box.x1 += (w - l) * 0.5;
//       box.y1 += (h - l) * 0.5;
//       box.x2 = box.x1 + l;
//       box.y2 = box.y1 + l;
//       box.score = score;
//       filterOutBoxes->push_back(box);
//     }
//   }
// }


// void OpencvMtcnn::generateBoundingBox(float* boxRegs,int i ,float* cls,float cur_sc_w, float cur_sc_h,const float threshold,std::vector<BoundingBox> & filterOutBoxes) 
// {
//   // clear output element
//   filterOutBoxes.clear();
//   int stride = 2;
//   int cellsize = 12;
//   int w = reg_w[i];
//   int h = reg_h[i];
//   // int n = box_shape[0];
//   for (int y = 0; y < h; y++) {
//     for (int x = 0; x < w; x++) {
//       float score = cls[0 * 2 * w * h + 1 * w * h + w * y + x];
//       if (score >= threshold) {
//         BoundingBox box;
//         box.dx1 = boxRegs[0 * w * h + w * y + x];
//         box.dy1 = boxRegs[1 * w * h + w * y + x];
//         box.dx2 = boxRegs[2 * w * h + w * y + x];
//         box.dy2 = boxRegs[3 * w * h + w * y + x];

//         box.x1 = floor((stride * x + 1) / cur_sc_w);
//         box.y1 = floor((stride * y + 1) / cur_sc_w);
//         box.x2 = floor((stride * x + cellsize) / cur_sc_w);
//         box.y2 = floor((stride * y + cellsize) / cur_sc_w);
//         box.score = score;
//         // add elements
//         filterOutBoxes.push_back(box);
//       }
//     }
//   }
// }

// void OpencvMtcnn::run_ONet(const cv::Mat &img, std::vector<BoundingBox> &rnet_boxes, std::vector<BoundingBox> &output_boxes)
// {
//     const int loop_num = rnet_boxes.size() / onet_n;
//     const int rem = rnet_boxes.size() % onet_n;
//     std::vector<cv::Mat> channels;
//     unsigned char* onet_cpu_input_tmp = (unsigned char*)onet_cpu_input[0];
//     for(int i = 0; i < onet_n;i++)
//     {
//         for(int j = 0; j < onet_c; j++)
//         {
//             cv::Mat channel(onet_h,onet_w,CV_8UC1,onet_cpu_input_tmp);
//             channels.push_back(channel);
//             onet_cpu_input_tmp+= onet_h*onet_w;
//         }
//     }
//     std::vector<cv::Mat> sample_norm_channels;
//     cv::split(img,sample_norm_channels);
//     cv::Mat alpha = cv::Mat(img.rows,img.cols,CV_8UC1,cv::Scalar(0));
//     sample_norm_channels.push_back(alpha);
//     std::vector<BoundingBox> batchBoxs;
//     vector<float> onet_cls_all, onet_regs_all,onet_landmarks_all;
//     for(int i = 0; i < loop_num; i++)
//     {
//         batchBoxs.clear();
//         for(int j = 0; j < onet_n; j++)
//         {
//             batchBoxs.push_back(total_rnet_boxes.at(j + i * onet_n));
//         }
//         buildInputChannels(sample_norm_channels, batchBoxs, cv::Size(48, 48),&channels);
//         int dim_order[4] = {0, 3, 1, 2};
//         int dim_input_order[4] = {0, 1, 2, 3};
//         int dim_input_transorder[4] = {0 , 2 , 3 , 1};
//         int dim_input_shape[4] = {onet_n,onet_c,onet_h,onet_w};
//         int dim_shape[4] = {onet_reg_n, onet_reg_h, onet_reg_w, 4};
//         int dim_shape_2[4] = {onet_reg_n, onet_reg_h, onet_reg_w, 2};
//         int dim_shape_3[4] = {onet_reg_n, onet_reg_h, onet_reg_w, 10};

//         cnrtTransDataOrder(onet_cpu_input[0],CNRT_UINT8,onet_input_trans,4,dim_input_shape,dim_input_transorder);

//         cnrtMemcpy(onet_mlu_input[0],onet_input_trans,onet_n * onet_c * onet_w * onet_h,CNRT_MEM_TRANS_DIR_HOST2DEV);
//         onet_model_infer.Run(onet_mlu_input,onet_mlu_output);
//         onet_model_mem_op.MemcpyOutputD2H(onet_cpu_output,onet_mlu_output);

//         cnrtTransOrderAndCast(onet_cpu_output[0],CNRT_FLOAT32,onet_reg_cout_trans,CNRT_FLOAT32,NULL,4,dim_shape,dim_order);
//         cnrtTransOrderAndCast(onet_cpu_output[1],CNRT_FLOAT32,onet_landmark_trans,CNRT_FLOAT32,NULL,4,dim_shape_2,dim_order);
//         cnrtTransOrderAndCast(onet_cpu_output[2],CNRT_FLOAT32,onet_confidence_trans,CNRT_FLOAT32,NULL,4,dim_shape_3,dim_order);

//         const float *begin0 = onet_reg_cout_trans;
//         const float *end0 = onet_reg_n * onet_reg_h * onet_reg_c * onet_reg_w+ begin0;
//         vector<float> onet_regs(begin0, end0);
//         onet_regs_all.insert(onet_regs_all.end(), onet_regs.begin(),onet_regs.end());

//         const float *begin1 =onet_confidence_trans;
//         const float *end1 =  onet_reg_n * onet_reg_h * 2 * onet_reg_w + begin1;
//         vector<float> onet_cls(begin1, end1);
//         onet_cls_all.insert(onet_cls_all.end(), onet_cls.begin(), onet_cls.end());

//         const float *begin2 = onet_landmark_trans;
//         const float *end2 =  onet_reg_n * onet_reg_h * 10 * onet_reg_w + begin2;
//         vector<float> onet_landmark(begin2, end2);
//         onet_landmarks_all.insert(onet_landmarks_all.end(), onet_landmark.begin(), onet_landmark.end());

//     }
//     if (rem > 0)
//     {
//       batchBoxs.clear();
//       for (int j = 0; j < rem; j++) 
//       {
//         batchBoxs.push_back(rnet_boxes.at(j + loop_num * rnet_n));
//       }
//       for (int j = rem; j < onet_n; j++) 
//       {
//         batchBoxs.push_back(rnet_boxes.at(rnet_boxes.size() - 1));
//       }
//         buildInputChannels(sample_norm_channels, batchBoxs, cv::Size(48, 48),&channels);
//         int dim_order[4] = {0, 3, 1, 2};
//         int dim_input_order[4] = {0, 1, 2, 3};
//         int dim_input_transorder[4] = {0 , 2 , 3 , 1};
//         int dim_input_shape[4] = {onet_n,onet_c,onet_h,onet_w};
//         int dim_shape[4] = {onet_reg_n, onet_reg_h, onet_reg_w, 4};
//         int dim_shape_2[4] = {onet_reg_n, onet_reg_h, onet_reg_w, 10};
//         int dim_shape_3[4] = {onet_reg_n, onet_reg_h, onet_reg_w, 2};

//         cnrtTransDataOrder(onet_cpu_input[0],CNRT_UINT8,onet_input_trans,4,dim_input_shape,dim_input_transorder);

//         cnrtMemcpy(onet_mlu_input[0],onet_input_trans,onet_n * onet_c * onet_w * onet_h,CNRT_MEM_TRANS_DIR_HOST2DEV);
//         onet_model_infer.Run(onet_mlu_input,onet_mlu_output);
//         onet_model_mem_op.MemcpyOutputD2H(onet_cpu_output,onet_mlu_output);

//         cnrtTransOrderAndCast(onet_cpu_output[0],CNRT_FLOAT32,onet_reg_cout_trans,CNRT_FLOAT32,NULL,4,dim_shape,dim_order);
//         cnrtTransOrderAndCast(onet_cpu_output[1],CNRT_FLOAT32,onet_landmark_trans,CNRT_FLOAT32,NULL,4,dim_shape_2,dim_order);
//         cnrtTransOrderAndCast(onet_cpu_output[2],CNRT_FLOAT32,onet_confidence_trans,CNRT_FLOAT32,NULL,4,dim_shape_3,dim_order);

//         const float *begin0 = onet_reg_cout_trans;
//         const float *end0 = onet_reg_n * onet_reg_h * onet_reg_c * onet_reg_w+ begin0;
//         vector<float> onet_regs(begin0, end0);
//         onet_regs_all.insert(onet_regs_all.end(), onet_regs.begin(),onet_regs.end());

//         const float *begin1 =onet_confidence_trans;
//         const float *end1 =  onet_reg_n * onet_reg_h * 2 * onet_reg_w + begin1;
//         vector<float> onet_cls(begin1, end1);
//         onet_cls_all.insert(onet_cls_all.end(), onet_cls.begin(), onet_cls.end());

//         const float *begin2 = onet_landmark_trans;
//         const float *end2 =  onet_reg_n * onet_reg_h * 10 * onet_reg_w + begin2;
//         vector<float> onet_landmark(begin2, end2);
//         onet_landmarks_all.insert(onet_landmarks_all.end(), onet_landmark.begin(), onet_landmark.end());

//     }
//     output_boxes.clear();
//     filteroutBoundingBox(rnet_boxes, onet_regs_all, onet_cls_all,onet_landmarks_all,std::vector<int>(), onet_threshold_, &output_boxes);
//     nms(&output_boxes, 0.7, 0, &rnet_boxes);
//     output_boxes.clear();
//     if (rnet_boxes.size() > 0) {
//       vector<BoundingBox> globalFilterBoxes(rnet_boxes);
//       rnet_boxes.clear();
//       for (int i = 0; i < globalFilterBoxes.size(); i++) 
//       {
//         if (globalFilterBoxes.at(i).x1 > 0 &&
//             globalFilterBoxes.at(i).x1 <
//                 globalFilterBoxes.at(i).x2 - m_min_size &&
//             globalFilterBoxes.at(i).y1 > 0 &&
//             globalFilterBoxes.at(i).y1 <
//                 globalFilterBoxes.at(i).y2 - m_min_size &&
//             globalFilterBoxes.at(i).x2 < timg_width &&
//             globalFilterBoxes.at(i).y2 < timg_height) {
//           output_boxes.push_back(globalFilterBoxes.at(i));
//         }
//       }
//     }
  

// }

// void OpencvMtcnn::nmsGlobal(std::vector<BoundingBox>& totalBoxes) 
// {
//   int min_size = m_min_size;
//   int img_W = timg_width;
//   int img_H = timg_height;
//   if (totalBoxes.size() > 0) {
//     vector<BoundingBox> globalFilterBoxes;
//     nms(&totalBoxes, 0.7, 1, &globalFilterBoxes);
//     totalBoxes.clear();
//     for (int i = 0; i < globalFilterBoxes.size(); i++) {
//       float regw = globalFilterBoxes[i].y2 - globalFilterBoxes[i].y1;
//       float regh = globalFilterBoxes[i].x2 - globalFilterBoxes[i].x1;
//       BoundingBox box;
//       float x1 = globalFilterBoxes[i].x1 + globalFilterBoxes[i].dy1 * regw;
//       float y1 = globalFilterBoxes[i].y1 + globalFilterBoxes[i].dx1 * regh;
//       float x2 = globalFilterBoxes[i].x2 + globalFilterBoxes[i].dy2 * regw;
//       float y2 = globalFilterBoxes[i].y2 + globalFilterBoxes[i].dx2 * regh;
//       float h = y2 - y1;
//       float w = x2 - x1;
//       float l = std::max(h, w);
//       x1 += (w - l) * 0.5;
//       y1 += (h - l) * 0.5;
//       x2 = x1 + l;
//       y2 = y1 + l;
//       box.x1 = x1, box.x2 = x2, box.y1 = y1, box.y2 = y2;
//       if (box.x1 > 0 && box.x1 < box.x2 - min_size && box.y1 > 0 &&
//           box.y1 < box.y2 - min_size && box.x2 < img_W && box.y2 < img_H) {
//         totalBoxes.push_back(box);
//       }
//     }
//   }
// }
// OpencvMtcnn::~OpencvMtcnn()
// {
//     for(int i = 0; i < pnet_model_num;i++)
//     {
//       pnet_model_mem_op[i].FreeCpuInput(pnet_cpu_input[i]);
//       pnet_model_mem_op[i].FreeCpuOutput(pnet_cpu_output[i]);
//       pnet_model_mem_op[i].FreeMluInput(pnet_mlu_input[i]);
//       pnet_model_mem_op[i].FreeMluOutput(pnet_mlu_output[i]);
//       delete reg_cout_trans[i];
//       delete confidence_trans[i];
//     }
//     rnet_model_mem_op.FreeCpuInput(rnet_cpu_input);
//     rnet_model_mem_op.FreeCpuOutput(rnet_cpu_output);
//     rnet_model_mem_op.FreeMluInput(rnet_mlu_input);
//     rnet_model_mem_op.FreeMluOutput(rnet_mlu_output);
//     delete rnet_input_trans;
//     delete rnet_reg_cout_trans;
//     delete rnet_confidence_trans;
//     onet_model_mem_op.FreeCpuInput(onet_cpu_input);
//     onet_model_mem_op.FreeCpuOutput(onet_cpu_output);
//     onet_model_mem_op.FreeMluInput(onet_mlu_input);
//     onet_model_mem_op.FreeMluOutput(onet_mlu_output);
//     delete onet_input_trans;
//     delete onet_reg_cout_trans;
//     delete onet_confidence_trans;
//     delete onet_landmark_trans;
// }

// void OpencvMtcnn::detect(cv::Mat &img, std::vector<face_box> &face_list) 
// {
//     cv::Mat working_img;
//     working_img = img.clone();
//     const int ih = working_img.rows;
//     const int iw = working_img.cols;

//     const int h = img_height;
//     const int w = img_width;

//     // float scale = 0;
//     if (ih > h || iw > w) 
//     {
//       scale = std::min(w * 1.0 / iw, h * 1.0 / ih);
//     } else {
//       // Note: do not remove this code, elsewise meanAp will drop 2%
//       scale = std::min(w / iw, h / ih);
//     }
//     int nw = static_cast<int>(iw * scale);
//     int nh = static_cast<int>(ih * scale);
//     dx = (w - nw) / 2;
//     dy = (h - nh) / 2;

//     cv::Mat im(cv::Size(w, h), CV_8UC3, cv::Scalar::all(128));

//     cv::Mat dstroi = im(cv::Rect(dx, dy, nw, nh));
//     cv::Mat resized;
//     cv::resize(working_img, resized, cv::Size(nw, nh));
//     resized.copyTo(dstroi);

//     im = im.t();

//     cv::cvtColor(im, im, cv::COLOR_BGR2RGB);

//     int img_h = im.rows;
//     int img_w = im.cols;


//     for (unsigned int i = 0; i < pnet_model_num; i++) 
//     {
//         std::vector<BoundingBox> boxes;
//         run_PNet(im, i, boxes);
//         total_pnet_boxes.insert(total_pnet_boxes.end(), boxes.begin(), boxes.end());
        
//     }
//     nmsGlobal(total_pnet_boxes);
    
// 	if(total_pnet_boxes.size()==0)
// 		return;
    
//     run_RNet(im,total_pnet_boxes,total_rnet_boxes);
//     run_ONet(im,total_rnet_boxes, total_onet_boxes);

//     for (int i = 0; i < total_onet_boxes.size(); i++)
//     {
//         face_box box;
//         std::swap(total_onet_boxes[i].x1, total_onet_boxes[i].y1);
//         std::swap(total_onet_boxes[i].x2, total_onet_boxes[i].y2);
//         box.x0 = static_cast<int>((total_onet_boxes[i].x1 - dx) / scale);
//         box.y0 = static_cast<int>((total_onet_boxes[i].y1 - dy) / scale);
//         box.x1 = static_cast<int>((total_onet_boxes[i].x2 - dx) / scale);
//         box.y1 = static_cast<int>((total_onet_boxes[i].y2 - dy) / scale);

//         for (int k = 0; k < 5; k++) 
//         {
//             std::swap(total_onet_boxes[i].points_x[k], total_onet_boxes[i].points_y[k]);
//             box.landmark.x[k] = static_cast<int>((total_onet_boxes[i].points_x[k] - dx) / scale);
//             box.landmark.y[k] = static_cast<int>((total_onet_boxes[i].points_y[k] - dy) / scale);
//         }
//         face_list.push_back(box);
//     }
// }



class HDDetectYolov3Private
{
     public:
        inline void AdjustRectPos(cv::Rect& Rect, const cv::Rect& RegionRect)
        { // no size change unless too big
            assert((Rect.width >= 0) && (Rect.height >= 0));
            assert((RegionRect.width >= 0) && (RegionRect.height >= 0));
            if (Rect.width > RegionRect.width)
                Rect.width = RegionRect.width;
            if (Rect.x < RegionRect.x)
                Rect.x = RegionRect.x;
            else if (Rect.x > RegionRect.width + RegionRect.x - Rect.width)
                Rect.x = RegionRect.width + RegionRect.x - Rect.width;
            if (Rect.height > RegionRect.height)
                Rect.height = RegionRect.height;
            if (Rect.y < RegionRect.y)
                Rect.y = RegionRect.y;
            else if (Rect.y > RegionRect.height + RegionRect.y - Rect.height)
                Rect.y = RegionRect.height + RegionRect.y - Rect.height;
            //assert(IsRectWithinRegion(Rect, RegionRect));
        }
        HDDetectYolov3Private(std::string weights, std::string func_name, int yoloType,int gpuid = 0)
        {
            yoloDetect = std::make_shared<Detection>(weights,func_name,gpuid,yoloType);
        }
        bool bbOverlap(const cv::Rect& box1, const cv::Rect& box2)
        {
            cv::Rect tmp = box1 & box2;
            return tmp.area() == box1.area();
            //return tmp.area();
        }
        cv::Size GetSize()
        {
            return yoloDetect->GetSize();
        }
        int getBatch()
        {
            return yoloDetect->GetBatch();
        }
        std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo)
        {
            auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
            if (x1min > x2min)
            {
                std::swap(x1min, x2min);
                std::swap(x1max, x2max);
            }
            return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
        };
        auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float {
            float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
            float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
            float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
            float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
            float overlap2D = overlapX * overlapY;
            float u = area1 + area2 - overlap2D;
            return u == 0 ? 0 : overlap2D / u;
        };

        std::stable_sort(binfo.begin(), binfo.end(),
                        [](const BBoxInfo& b1, const BBoxInfo& b2) { return b1.prob > b2.prob; });
        std::vector<BBoxInfo> out;
        for (auto i : binfo)
        {
            bool keep = true;
            for (auto j : out)
            {
                if (keep)
                {
                    float overlap = computeIoU(i.box, j.box);
                    keep = overlap <= nmsThresh;
                }
                else
                    break;
            }
            if (keep) out.push_back(i);
        }
            return out;
        }
        

        void detect(std::vector<cv::Mat>& img, std::vector<std::vector<DetectedObject>>& res, std::vector<cv::Rect> vecRoi, int area)
        {
            std::vector<cv::Size> inputSize{cv::Size(img[0].cols,img[0].rows)};
            yoloDetect->Detect(img,res,inputSize);

            if (vecRoi.size() == 0)
            {
                return;
            }
            for (auto& v : vecRoi)
            {
                //调整框面积过小
                if (v.area() < 10000)
                {
                    int xmin = v.x;
                    int ymin = v.y;
                    int width = v.width;
                    int height = v.height;
                    int center_x = (int)((xmin + width + xmin) / 2);
                    int center_y = (int)((ymin + ymin + height) / 2);
                    cv::Size inputsize = yoloDetect->GetSize();
                    int inputwidth = inputsize.width;
                    int inputheight = inputsize.height;
                    xmin = center_x - (int)(inputwidth / 2);
                    ymin = center_y - (int)(inputheight / 2);
                    width = inputwidth;
                    height = inputheight;
                    v.x = xmin;
                    v.y = ymin;
                    v.width = width;
                    v.height = height;
                }

                //调整画框比例
                int width = v.width;
                int height = v.height;
                int ratio =(float)(width / height) < 1 ? 1 / (float)(width / height) : (float)(width / height);
                if(ratio >= 1.8)
                {
                    int area = v.area();
                    if(width > height)
                    {
                        v.height = sqrt((float)(area / 1.4));
                        v.width = v.height * 1.4;
                    }
                    else
                    {
                        v.width = sqrt((float)(area / 1.4));
                        v.height = v.width * 1.4;
                    }
                }
            }
            //cv::Mat originImgCopy = img[0].clone();
            //对预选框进行调整
            for (int i = 0; i < vecRoi.size(); i++)
            {
                AdjustRectPos(vecRoi[i], cv::Rect(0, 0, img[0].cols, img[0].rows));
            }

            //对roi进行一一检测
            for (int i = 0; i < vecRoi.size(); i++)
            {
                cv::Rect roiTmp = vecRoi[i];
                cv::Mat imgRoi = img[0](roiTmp).clone();
                std::vector<cv::Mat> vBatchImages;
                vBatchImages.push_back(imgRoi);
                std::vector<std::vector<DetectedObject>> result;
              
                yoloDetect->Detect(vBatchImages, result,inputSize);

                processDetectResult(result, cv::Rect(0, 0, imgRoi.cols, imgRoi.rows), area);
                for (auto& v : result)
                {
                    for (auto& r : v)
                    {
                        //cv::rectangle(imgRoi, r.bounding_box, cv::Scalar(255, 255, 0), 2);
                        r.bounding_box.x = r.bounding_box.x + roiTmp.x;
                        r.bounding_box.y = r.bounding_box.y + roiTmp.y;
                    }
                }
                res[0].insert(res[0].end(), result[0].begin(), result[0].end());
            }
            NmsAll(res);
        }
        void detect2(std::vector<MatAndRois>& matRoiInputs,std::vector<std::vector<DetectedObject>> &res)
        {
            assert(matRoiInputs.size() == 1);
            cv::Mat resizeimg = matRoiInputs[0].input;
            cv::Size img_size = matRoiInputs[0].sizeInput;
            std::vector<cv::Rect> rects = matRoiInputs[0].rectRois;
            std::vector<cv::Mat> roisinput = matRoiInputs[0].matRois;
            assert(!(resizeimg.empty()));
            for(int i = 0; i < rects.size(); i++)
            {
                assert(rects[i].empty() == roisinput[i].empty());
            }
            vector<cv::Size> img_sizes;
            img_sizes.push_back(img_size);
            std::vector<cv::Mat>trtInput{resizeimg};
            yoloDetect->Detect(trtInput, res, img_sizes);
         
            if(rects.size() == 0)
            {
                return;
            }
            for (int i = 0; i < rects.size(); i++)
            {
                cv::Rect roiTmp = rects[i];
                std::vector<std::vector<DetectedObject>> result;
                cv::Mat roiresizeimg = roisinput[i];
                std::vector<cv::Size> img_sizes;
                img_sizes.push_back(cv::Size(roiTmp.width,roiTmp.height));
                std::vector<cv::Mat> imgVec{roiresizeimg}; 
                yoloDetect->Detect(imgVec, result, img_sizes);
              
                int area = 3600;
                processDetectResult(result, cv::Rect(0, 0, rects[i].width, rects[i].height), area);
                for (auto& v : result)
                {
                    for (auto& r : v)
                    {
                        //cv::rectangle(imgRoi, r.bounding_box, cv::Scalar(255, 255, 0), 2);
                        r.bounding_box.x = r.bounding_box.x + roiTmp.x;
                        r.bounding_box.y = r.bounding_box.y + roiTmp.y;
                    }
                }
                res[0].insert(res[0].end(), result[0].begin(), result[0].end());
                //cv::imwrite("/data5/ld/yolo/test/test_hdtrackyolov3/nmspre/" + std::to_string(i) + ".jpg", imgRoi);
            }
            NmsAll(res);
        }
        void NmsAll(std::vector<std::vector<DetectedObject>>& vec_result)
        {
            assert(vec_result.size() == 1);
            //std::cout << "nms size :" << vec_result[0].size() << std::endl;
            std::vector<BBoxInfo> bInfo;
            std::vector<DetectedObject> tmpBatchResult;
            for (int i = 0; i < vec_result[0].size(); i++)
            {
                BBoxInfo tmp;
                //BBoxzhn tmpP;
                tmp.box.x1 = vec_result[0][i].bounding_box.x;
                tmp.box.y1 = vec_result[0][i].bounding_box.y;
                tmp.box.x2 = vec_result[0][i].bounding_box.x + vec_result[0][i].bounding_box.width;
                tmp.box.y2 = vec_result[0][i].bounding_box.y + vec_result[0][i].bounding_box.height;
                tmp.label = vec_result[0][i].object_class;
                tmp.prob = vec_result[0][i].prob;
                bInfo.push_back(tmp);
            }
            vec_result.clear();
            //        auto remaining = nmsAllClasseszhn(_p_net->getNMSThresh(),
            //                                          bInfo,
            //                                          _p_net->getNumClasses(),
            //                                          _vec_net_type[_config.net_type]);
            auto remaining = nonMaximumSuppression(0.45, bInfo);
            //std::cout << "nms out size :" << remaining.size() << std::endl;
            for (const auto& b : remaining)
            {
                DetectedObject res;
                res.object_class = b.label;
                res.prob = b.prob;
                int x = b.box.x1;
                int y = b.box.y1;
                int w = b.box.x2 - b.box.x1;
                int h = b.box.y2 - b.box.y1;
                res.bounding_box = cv::Rect(x, y, w, h);
                tmpBatchResult.push_back(res);
            }

            vec_result.push_back(tmpBatchResult);
        }
        void processDetectResult(std::vector<std::vector<DetectedObject>>& res, cv::Rect rectRoi, int area)
        {
            assert(res.size() == 1);
            for (auto it = res[0].begin(); it != res[0].end();)
            {
                // std::cout << "rect: " << it->bounding_box << std::endl;

                if (GetRoiRemove(it->bounding_box, rectRoi, area))
                {
                    res[0].erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
        //去除在边界的目标框和去除较大的框
        bool GetRoiRemove(const cv::Rect box1, const cv::Rect box2, int area)
        {
            //框的长和宽占到roi的1/4则删除
//            if(box1.width > box2.width/4 || box1.height > box2.height/4 )
//            {
//                return true;
//            }
            if ((int)box1.area() > area)
            {
                return true;
            }
            int xmin = box1.x;
            int ymin = box1.y;
            int xmax = box1.x + box1.width;
            int ymax = box1.y + box1.height;
            if (xmin < 5 || ymin < 5 || abs(box2.x + box2.width - xmax) < 5 || abs(box2.y + box2.height - ymax) < 5)
            {
                return true;
            }
            return false;

        }

    private:
        std::shared_ptr<Detection> yoloDetect;
};

void HDDetectYolov3::adjustRect(std::vector<cv::Rect>& rects,cv::Size sizeInput)
{
    for (auto & rect:rects)
    {
        m_pHandleHDDetectYolov3Private->AdjustRectPos(rect,cv::Rect(0,0,sizeInput.width,sizeInput.height));
    }
    for (auto& v : rects)
    {
        //调整框面积过小
        if (v.area() < 10000)
        {
            int xmin = v.x;
            int ymin = v.y;
            int width = v.width;
            int height = v.height;
            int center_x = (int)((xmin + width + xmin) / 2);
            int center_y = (int)((ymin + ymin + height) / 2);
            cv::Size inputsize = GetSize();
            int inputwidth = inputsize.width;
            int inputheight = inputsize.height;
            xmin = center_x - (int)(inputwidth / 2);
            ymin = center_y - (int)(inputheight / 2);
            width = inputwidth;
            height = inputheight;
            v.x = xmin;
            v.y = ymin;
            v.width = width;
            v.height = height;
        }

        //调整画框比例
        float width = v.width;
        float height = v.height;
        float ratio = width / height < 1 ? 1.0 / (width / height) : width / height;
        if(ratio >= 1.8)
        {
            int area = v.area();
            if(width > height)
            {
                v.height = sqrt((float)(area / 1.4));
                v.width = v.height * 1.4;
            }
            else
            {
                v.width = sqrt((float)(area / 1.4));
                v.height = v.width * 1.4;
            }
        }
    }
    for (int i = 0; i < rects.size(); i++)
    {
        m_pHandleHDDetectYolov3Private->AdjustRectPos(rects[i], cv::Rect(0, 0, sizeInput.width, sizeInput.height));
    }
}

HDDetectYolov3::HDDetectYolov3(std::string weights, std::string func_name, int yoloType ,int gpuid, float thresh )
{
    m_pHandleHDDetectYolov3Private = std::make_shared<HDDetectYolov3Private>(weights, func_name, yoloType,gpuid);
}
void HDDetectYolov3::detect(std::vector<cv::Mat>& img, std::vector<std::vector<DetectedObject>>& res, std::vector<cv::Rect> vecRoi, int area)
{
    m_pHandleHDDetectYolov3Private->detect(img, res, vecRoi, area);
}
void HDDetectYolov3::detect2(std::vector<MatAndRois>& matRoiInputs,std::vector<std::vector<DetectedObject>> &res)
{
    m_pHandleHDDetectYolov3Private->detect2(matRoiInputs, res);
}
cv::Size HDDetectYolov3::GetSize()
{
    return m_pHandleHDDetectYolov3Private->GetSize();
}
int HDDetectYolov3::GetBatch()
{
    return m_pHandleHDDetectYolov3Private->getBatch();
}



