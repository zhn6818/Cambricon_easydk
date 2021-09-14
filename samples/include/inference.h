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

#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <string>
#include <utility>
#include <vector>
#include "cncv.h"
#include "cnrt.h"

#include "cnpostproc.h"
#include "device/mlu_context.h"
#include "easybang/resize_and_colorcvt.h"
#include "easycodec/easy_decode.h"
#include "easyinfer/easy_infer.h"
#include "easyinfer/mlu_memory_op.h"
#include "easyinfer/model_loader.h"
#include "easytrack/easy_track.h"
#include <deque>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

typedef std::pair<std::string, float> Prediction;

#define SAFECALL(func, expect) \
  do { \
    auto ret = (func); \
    if (ret != (expect)) { \
      std::cerr << "Call " << #func << "failed, error code: " << ret << std::endl; \
      abort(); \
    } \
  } while(0)

#define CNRT_SAFECALL(func) SAFECALL(func, CNRT_RET_SUCCESS)
#define CNCV_SAFECALL(func) SAFECALL(func, CNCV_STATUS_SUCCESS)

class Detection{
public:
  Detection(const std::string &model_path, const std::string &func_name, const int device_id);
  void Preprocess(const std::vector<cv::Mat>& imgs, int dst_w, int dst_h, void* output);
  void testInput(void* in);
  void KeepAspectRatio(cncvRect* dst_roi,
                     const cncvImageDescriptor& src,
                     const cncvImageDescriptor& dst);
  void Detect(std::vector<cv::Mat> &preprocessedImages,
              std::vector<std::vector<DetectedObject>> &arrDetection, std::vector<cv::Size> &sizeDetect);
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
  void* model_input[1];
  void* model_output[1];
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
  void** cpu_src_imgs;
  void** cpu_dst_imgs;
  void** cpu_tmp_imgs;

  void** mlu_input;
  void** mlu_output;
  void** tmp;

  void** cpu_output_{nullptr};

};

class Classifycation{
public:

  Classifycation(const std::string &model_path, const std::string &func_name, const std::string &names, const int device_id);
  ~Classifycation();
  void Preprocess(const std::vector<cv::Mat>& imgs, int dst_w, int dst_h, void* output);
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
  void** cpu_src_imgs;
  void** cpu_tmp_imgs;
  void** cpu_dst_imgs;

  void** mlu_input;
  void** mlu_output;
  void** tmp;


  edk::MluMemoryOp mem_op_;
  edk::EasyInfer infer_;
  edk::MluResizeConvertOp rc_op_;
  edk::MluContext env_;
  std::shared_ptr<edk::ModelLoader> model_{nullptr};
  std::unique_ptr<edk::CnPostproc> postproc_{nullptr};
  
  uint32_t batch_size;
  uint32_t cpu_src_imgs_buff_size, dst_size;
  void* model_input[1];
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

  void** cpu_output_{nullptr};

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
  void getfeat(std::vector<cv::Mat> &vBatchImages, cv::Mat& feat);
  void processFeat_test(const cv::Mat& feat, std::vector<cv::Point> pts, const cv::Size& srcImageSize, float smoke_thresh, std::vector<std::vector<cv::Point>>& contours, bool& isSmoke, int& binary_pixels, std::vector<std::vector<cv::Point>>& contours_all);
  void processFeat(const cv::Mat &feat, std::vector<cv::Point> pts, const cv::Size& srcImageSize, float smoke_thres, std::vector<std::vector<cv::Point>>& contours, bool& isSmoke);


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
