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

#ifndef EDK_SAMPLES_STREAM_APP_DETECTION_RUNNER_H_
#define EDK_SAMPLES_STREAM_APP_DETECTION_RUNNER_H_


#include <memory>
#include <string>


#include "cnpostproc.h"
#include "device/mlu_context.h"
#include "easybang/resize_and_colorcvt.h"
#include "easycodec/easy_decode.h"
#include "easyinfer/easy_infer.h"
#include "easyinfer/mlu_memory_op.h"
#include "easyinfer/model_loader.h"
#include "easytrack/easy_track.h"
#include <opencv2/core/core.hpp>
#include <mutex>
#include <utility>
#include <vector>


// class DetectionRunner : public StreamRunner {
class DetectionRunner {
 public:
  DetectionRunner(const std::string& model_path, const std::string& func_name,const int device_id);
  ~DetectionRunner();

  void Detect(std::vector<std::vector<cv::Mat>> &preprocessedImages, std::vector<std::vector<DetectedObject>> &arrDetection,std::vector<cv::Size> &image_paras);
  static void Pre(std::vector<cv::Mat> &vBatchImages,std::vector<std::vector<cv::Mat>> &preprocessedImages,std::vector<cv::Size> &image_paras);
  int GetBatch();
  
 private:
  static void WrapInputLayer(int batchsize,std::vector<std::vector<cv::Mat>>* wrappedImages);
  static void Preprocess(const std::vector<cv::Mat>& sourceImages,std::vector<std::vector<cv::Mat>>* destImages);
  edk::MluMemoryOp mem_op_;
  edk::EasyInfer infer_;
  edk::MluResizeConvertOp rc_op_;
  edk::MluContext env_;
  std::shared_ptr<edk::ModelLoader> model_{nullptr};
  std::unique_ptr<edk::CnPostproc> postproc_{nullptr};
  std::unique_ptr<edk::FeatureMatchTrack> tracker_{nullptr};
  void **mlu_output_{nullptr}, **cpu_output_{nullptr}, **mlu_input_{nullptr};

  // bool show_;
  // bool save_video_;
  int net_w;
  int net_h;
  int net_n;
  int net_c;
  int out_n;
  int out_c;
  int out_w;
  int out_h;
  // int image_width;
  // int image_height;
  void** input_data;//输入数据，在cpu上的指针
  void** cpuData_;
  void** cpuTrans_;
  void **firstConvData_;
  void **inputMluPtrS_infer;
  void **inputMluTempPtrS_infer;
  void ** mluData_infer;
  void ** cpuTempData_infer;


  void **outputMluPtrS_infer;
  void **outputMluTempPtrS_infer;
  std::vector<void*> outCpuPtrs_;
  std::vector<void*> outTrans_;
  int outCount_[2];
};

#endif  // EDK_SAMPLES_STREAM_APP_DETECTION_RUNNER_H_

