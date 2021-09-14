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

cv::Size Detection::GetSize(){ return cv::Size(net_w, net_h); }

void Detection::KeepAspectRatio(cncvRect* dst_roi, const cncvImageDescriptor& src, const cncvImageDescriptor& dst) {
  float src_ratio = static_cast<float>(src.width) / src.height;
  float dst_ratio = static_cast<float>(dst.width) / dst.height;
  if (src_ratio < dst_ratio) {
    int pad_lenth = dst.width - src_ratio * dst.height;
    pad_lenth = (pad_lenth % 2) ? pad_lenth - 1 : pad_lenth;
    if (dst.width - pad_lenth / 2 < 0) return;
    dst_roi->w = dst.width - pad_lenth;
    dst_roi->x = pad_lenth / 2;
    dst_roi->y = 0;
    dst_roi->h = dst.height;
  } else if (src_ratio > dst_ratio) {
    int pad_lenth = dst.height - dst.width / src_ratio;
    pad_lenth = (pad_lenth % 2) ? pad_lenth - 1 : pad_lenth;
    if (dst.height - pad_lenth / 2 < 0) return;
    dst_roi->h = dst.height - pad_lenth;
    dst_roi->y = pad_lenth / 2;
    dst_roi->x = 0;
    dst_roi->w = dst.width;
  }
}


void Detection::Preprocess(const std::vector<cv::Mat>& imgs, int dst_w, int dst_h, void* output) {
  if (imgs.empty()) {
    std::cerr << "no image" << std::endl;
    abort();
  }
  // CNCV算子的batchsize是运行时可变的，有多少数据就设成对应的batchsize可以节省算力
  std::cout << "start preprocessing" << std::endl;
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

  void* workspace;
  size_t workspace_size;

  int src_w = imgs[0].cols;
  int src_h = imgs[0].rows;
  int src_stride = imgs[0].step;

  std::cout << "maxSize: " << maxSize << std::endl;
  assert(maxSize <= cpu_src_imgs_buff_size);
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
  CNRT_SAFECALL(cnrtMemcpy(mlu_input, cpu_src_imgs, batch_size * sizeof(void*), CNRT_MEM_TRANS_DIR_HOST2DEV));

  CNRT_SAFECALL(cnrtMemcpy(tmp, cpu_tmp_imgs, batch_size * sizeof(void*), CNRT_MEM_TRANS_DIR_HOST2DEV));

  // wrap output memory into dst pointer array
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    cpu_dst_imgs[idx] = reinterpret_cast<uint8_t*>(output) + idx * dst_size;
  }
  // copy mlu dst imgs pointer cpu array to mlu
  //   CNRT_SAFECALL(cnrtMalloc((void**)&mlu_output, batch_size * sizeof(void*)));
  CNRT_SAFECALL(cnrtMemcpy(mlu_output, cpu_dst_imgs, batch_size * sizeof(void*), CNRT_MEM_TRANS_DIR_HOST2DEV));

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

Detection::Detection(const std::string& model_path, const std::string& func_name, const int device_id) {
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
  auto& in_shape = model_->InputShape(0);
  auto& out_shape = model_->OutputShape(0);
  int outNum = (int)model_->OutputNum();

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

  postproc_.reset(new edk::Yolov3Postproc);
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

  cpu_src_imgs = new void*[batch_size];

  cpu_src_imgs_buff_size = 1 * sizeof(uint8_t);
  std::cout << "cpu_src_imgs_buff_size: " << cpu_src_imgs_buff_size << std::endl;

  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
  }

  CNRT_SAFECALL(cnrtMalloc((void**)&mlu_input, batch_size * sizeof(void*)));

  cpu_tmp_imgs = new void*[batch_size];

  dst_size = net_w * net_h * 4 * sizeof(uint8_t);
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    CNRT_SAFECALL(cnrtMalloc(&(cpu_tmp_imgs[idx]), dst_size));
  }

  CNRT_SAFECALL(cnrtMalloc((void**)&tmp, batch_size * sizeof(void*)));

  cpu_dst_imgs = new void*[batch_size];
  CNRT_SAFECALL(cnrtMalloc((void**)&mlu_output, batch_size * sizeof(void*)));

  cpu_output_ = mem_op_.AllocCpuOutput();
  // workspace_size = 0;

  std::cout << "initialize success" << std::endl;
}

void Detection::Detect(std::vector<cv::Mat>& preprocessedImages, std::vector<std::vector<DetectedObject>>& arrDetection, std::vector<cv::Size> &sizeDetect)

{
  // env_.SetDeviceId(device);
  std::cout << "test";
  // paras.resize(0);
  env_.BindDevice();
  std::cout << "BindDevice " << std::endl;
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

  mem_op_.MemcpyOutputD2H(cpu_output_, (void**)model_output);
  std::vector<std::vector<edk::DetectObject>> objDetection;
  objDetection.clear();
  std::vector<std::pair<float*, uint64_t>> postproc_param;
  postproc_param.push_back(
      std::make_pair(reinterpret_cast<float*>(cpu_output_[0]), model_->OutputShape(0).BatchDataCount()));
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

Classifycation::~Classifycation() {
  if (nullptr != model_output) mem_op_.FreeMluOutput(model_output);
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
Classifycation::Classifycation(const std::string& model_path, const std::string& func_name, const std::string& names,
                               const int device_id) {
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
  auto& in_shape = model_->InputShape(0);
  auto& out_shape = model_->OutputShape(0);
  int outNum = (int)model_->OutputNum();

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

  std::ifstream fin(names, std::ios::in);
  char line[1024] = {0};
  std::string name = "";
  while (fin.getline(line, sizeof(line))) {
    std::stringstream word(line);
    word >> name;
    std::cout << "name: " << name << std::endl;
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
  //model_output.AllocMluOutput()
  model_output = mem_op_.AllocMluOutput();

  CNRT_SAFECALL(cnrtCreateQueue(&queue));
  CNCV_SAFECALL(cncvCreate(&handle));
  CNCV_SAFECALL(cncvSetQueue(handle, queue));

  src_rois = new cncvRect[batch_size];
  tmp_rois = new cncvRect[batch_size];

  CNRT_SAFECALL(cnrtMalloc((void**)&mlu_input, batch_size * sizeof(void*)));
  CNRT_SAFECALL(cnrtMalloc((void**)&tmp, batch_size * sizeof(void*)));
  CNRT_SAFECALL(cnrtMalloc((void**)&mlu_output, batch_size * sizeof(void*)));

  cpu_src_imgs = new void*[batch_size];
  cpu_src_imgs_buff_size = 100 * sizeof(uint8_t);
  std::cout << "cpu_src_imgs_buff_size: " << cpu_src_imgs_buff_size << std::endl;
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
  }

  cpu_tmp_imgs = new void*[batch_size];

  dst_size = net_w * net_h * 4 * sizeof(uint8_t);
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    CNRT_SAFECALL(cnrtMalloc(&(cpu_tmp_imgs[idx]), dst_size));
  }

  cpu_dst_imgs = new void*[batch_size];

  cpu_output_ = mem_op_.AllocCpuOutput();

  std::cout << "initialize Classifycation success" << std::endl;
}

std::vector<std::vector<Prediction>> Classifycation::Classify(std::vector<cv::Mat>& vBatchImages, int N) {
  // std::cout << "test classify" << std::endl;
  env_.BindDevice();
  Preprocess(vBatchImages, net_w, net_h, model_input[0]);
  // std::cout << "prePare over" << std::endl;
  infer_.Run(model_input, model_output);
  // std::cout << "Run over" << std::endl;
  mem_op_.MemcpyOutputD2H(cpu_output_, model_output);
  // std::cout << "mem_op_ over" << std::endl;
  std::vector<std::vector<edk::DetectObject>> detect_result;
  std::vector<std::pair<float*, uint64_t>> postproc_param;
  postproc_param.push_back(
      std::make_pair(reinterpret_cast<float*>(cpu_output_[0]), model_->OutputShape(0).DataCount()));
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
void Classifycation::Preprocess(const std::vector<cv::Mat>& imgs, int dst_w, int dst_h, void* output) {
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

  void* workspace;
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

  CNRT_SAFECALL(cnrtMemcpy(mlu_input, cpu_src_imgs, batch_size * sizeof(void*), CNRT_MEM_TRANS_DIR_HOST2DEV));

  CNRT_SAFECALL(cnrtMemcpy(tmp, cpu_tmp_imgs, batch_size * sizeof(void*), CNRT_MEM_TRANS_DIR_HOST2DEV));
  for (uint32_t idx = 0; idx < batch_size; ++idx) {
    cpu_dst_imgs[idx] = reinterpret_cast<uint8_t*>(output) + idx * dst_size;
  }
  CNRT_SAFECALL(cnrtMemcpy(mlu_output, cpu_dst_imgs, batch_size * sizeof(void*), CNRT_MEM_TRANS_DIR_HOST2DEV));

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

DetectionRunner::DetectionRunner(const std::string& model_path, const std::string& func_name, const int device_id) {
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
  auto& in_shape = model_->InputShape(0);
  auto& out_shape = model_->OutputShape(0);
  int outNum = (int)model_->OutputNum();

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
  inputMluTempPtrS_infer = reinterpret_cast<void**>(malloc(sizeof(void*) * 1));
  inputMluTempPtrS_infer[0] = malloc(net_n * net_c * net_w * net_h);
  inputMluPtrS_infer = reinterpret_cast<void**>(new void*[1]);
  cnrtMalloc(&(inputMluPtrS_infer[0]), net_n * net_c * net_w * net_h);
  cpuData_ = new (void*);
  cpuData_[0] = new float[net_n * 3 * net_w * net_h];
  cpuTrans_ = new (void*);
  cpuTrans_[0] = new float[net_n * 3 * net_w * net_h];
  firstConvData_ = new (void*);
  firstConvData_[0] = new char[net_n * 3 * net_w * net_h];
}

DetectionRunner::~DetectionRunner() {
  if (nullptr != mlu_output_) mem_op_.FreeMluOutput(mlu_output_);
  if (nullptr != cpu_output_) mem_op_.FreeCpuOutput(cpu_output_);

  if (nullptr != inputMluPtrS_infer) mem_op_.FreeMluInput(inputMluPtrS_infer);

  if (nullptr != inputMluTempPtrS_infer) {
    free(inputMluTempPtrS_infer[0]);
    free(inputMluTempPtrS_infer);
  }
  delete[] reinterpret_cast<float*>(cpuData_[0]);
  delete cpuData_;
  delete[] reinterpret_cast<float*>(cpuTrans_[0]);
  delete cpuTrans_;
  delete[] reinterpret_cast<char*>(firstConvData_[0]);
  delete firstConvData_;
}

int DetectionRunner::GetBatch() { return net_n; }

void DetectionRunner::WrapInputLayer(int batchsize, std::vector<std::vector<cv::Mat>>* wrappedImages) {
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

void DetectionRunner::Preprocess(const std::vector<cv::Mat>& sourceImages,
                                 std::vector<std::vector<cv::Mat>>* destImages) {
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
          cv::Range((static_cast<float>(input_dim) - new_h) / 2, (static_cast<float>(input_dim) - new_h) / 2 + new_h),
          cv::Range((static_cast<float>(input_dim) - new_w) / 2, (static_cast<float>(input_dim) - new_w) / 2 + new_w)));
    }
    cv::Mat sample_float;
    sample_resized.convertTo(sample_float, CV_32FC3, 1);
    cv::split(sample_float, (*destImages)[i]);
    cv::Mat B_tmp = (*destImages)[i][0];
    (*destImages)[i][0] = (*destImages)[i][2];
    (*destImages)[i][2] = B_tmp;
  }
}

void DetectionRunner::Pre(std::vector<cv::Mat>& vBatchImages, std::vector<std::vector<cv::Mat>>& preprocessedImages,
                          std::vector<cv::Size>& image_paras) {
  int batchsize = vBatchImages.size();
  for (int i = 0; i < batchsize; i++) {
    cv::Size para(vBatchImages[i].cols, vBatchImages[i].rows);
    image_paras.push_back(para);
  }
  WrapInputLayer(batchsize, &preprocessedImages);
  Preprocess(vBatchImages, &preprocessedImages);
}

void DetectionRunner::Detect(std::vector<std::vector<cv::Mat>>& preprocessedImages,
                             std::vector<std::vector<DetectedObject>>& arrDetection,
                             std::vector<cv::Size>& image_paras) {
  float* data = reinterpret_cast<float*>(cpuData_[0]);
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
    void* temp_ptr = nullptr;

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
  std::vector<std::pair<float*, uint64_t>> postproc_param;
  postproc_param.push_back(
      std::make_pair(reinterpret_cast<float*>(cpu_output_[0]), model_->OutputShape(0).BatchDataCount()));
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

ClassificationRunner::ClassificationRunner(const std::string& model_path, const std::string& func_name,
                                           const std::string& names, const int device_id) {
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
  auto& in_shape = model_->InputShape(0);
  auto& out_shape = model_->OutputShape(0);
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
  mluData_infer = reinterpret_cast<void**>(malloc(sizeof(void*) * 1));
  cnrtMalloc(&(mluData_infer[0]), net_n * net_c * net_w * net_h);

  inputCpuPtrS = (void**)malloc(sizeof(void*) * 1);
  inputCpuPtrS[0] = (void*)malloc(net_w * net_h * net_c * net_n);
}

ClassificationRunner::~ClassificationRunner() {
  // Stop();
  if (nullptr != mlu_output_) mem_op_.FreeMluOutput(mlu_output_);
  if (nullptr != cpu_output_) mem_op_.FreeCpuOutput(cpu_output_);
  if (nullptr != mluData_infer) mem_op_.FreeMluInput(mluData_infer);
  if (nullptr != inputCpuPtrS) mem_op_.FreeCpuOutput(inputCpuPtrS);
}

int ClassificationRunner::GetBatch() { return net_n; }

void ClassificationRunner::Pre(const std::vector<cv::Mat>& vBatchImages) {
  unsigned char* ptr = (unsigned char*)inputCpuPtrS[0];
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

std::vector<std::vector<Prediction>> ClassificationRunner::Classify(std::vector<cv::Mat>& vBatchImages, int N) {
  // preprocess
  Pre(vBatchImages);

  // run inference
  infer_.Run(mluData_infer, mlu_output_);
  mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_);

  // post process
  std::vector<std::vector<edk::DetectObject>> detect_result;
  std::vector<std::pair<float*, uint64_t>> postproc_param;
  postproc_param.push_back(
      std::make_pair(reinterpret_cast<float*>(cpu_output_[0]), model_->OutputShape(0).DataCount()));
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

CrowdCountPredictor::CrowdCountPredictor(const std::string& model_path, const std::string& func_name,
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
  auto& in_shape = model_->InputShape(0);
  auto& out_shape = model_->OutputShape(0);
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
  if (nullptr != mlu_output_) mem_op_.FreeMluOutput(mlu_output_);
  if (nullptr != cpuData_) mem_op_.FreeCpuOutput(cpuData_);
  if (nullptr != cpu_output_) mem_op_.FreeCpuOutput(cpu_output_);
  if (nullptr != mlu_input_) mem_op_.FreeMluInput(mlu_input_);
}

int CrowdCountPredictor::GetBatch() { return net_n; }

void CrowdCountPredictor::Pre(cv::Mat& img) {
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

  float* data = reinterpret_cast<float*>(cpuData_[0]);
  const float* indata = normalized_std.ptr<float>(0);
  for (int i = 0; i < net_n * net_c * net_w * net_h; i++) {
    data[i] = (float)*indata++;
  }
}

void CrowdCountPredictor::run(cv::Mat& images, cv::Mat& result) {
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
  float* out = reinterpret_cast<float*>(cpu_output_[0]);
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

ResnetSegment::ResnetSegment(const std::string& model_path, const std::string& func_name, const int device_id) {
  std::cout << "Cambricion Resnet18 FPN segment inference" << std::endl;
  env_.SetDeviceId(device_id);
  env_.BindDevice();
  std::cout << "model path : " << model_path << std::endl;
  model_ = std::make_shared<edk::ModelLoader>(model_path.c_str(), func_name.c_str());
  mem_op_.SetModel(model_);
  // init easy_infer
  infer_.Init(model_, device_id);
  auto& in_shape = model_->InputShape(0);
  auto& out_shape = model_->OutputShape(0);
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

  mluData_infer = reinterpret_cast<void**>(malloc(sizeof(void*) * 1));
  cnrtMalloc(&(mluData_infer[0]), net_n * net_c * net_w * net_h);

  inputCpuPtrS = (void**)malloc(sizeof(void*) * 1);
  inputCpuPtrS[0] = (void*)malloc(net_w * net_h * net_c * net_n);
}

ResnetSegment::~ResnetSegment() {
  if (nullptr != mlu_output_) mem_op_.FreeMluOutput(mlu_output_);
  if (nullptr != cpu_output_) mem_op_.FreeCpuOutput(cpu_output_);
  if (nullptr != mluData_infer) mem_op_.FreeMluInput(mluData_infer);
  if (nullptr != inputCpuPtrS) mem_op_.FreeCpuOutput(inputCpuPtrS);
}
void ResnetSegment::Pre(const std::vector<cv::Mat>& vBatchImages) {
  unsigned char* ptr = (unsigned char*)inputCpuPtrS[0];
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

void ResnetSegment::getfeat(std::vector<cv::Mat>& vBatchImages, cv::Mat& feat) {
  Pre(vBatchImages);
  infer_.Run(mluData_infer, mlu_output_);
  mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_);

  float* out = reinterpret_cast<float*>(cpu_output_[0]);

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
void ResnetSegment::processFeat(const cv::Mat& feat, std::vector<cv::Point> pts, const cv::Size& srcImageSize,
                                float smoke_thres, std::vector<std::vector<cv::Point>>& contours, bool& isSmoke) {
  int binary_pixels = 0;
  std::vector<std::vector<cv::Point>> contours_all;
  processFeat_test(feat, pts, srcImageSize, smoke_thres, contours, isSmoke, binary_pixels, contours_all);
}

void ResnetSegment::processFeat_test(const cv::Mat& feat, std::vector<cv::Point> pts, const cv::Size& srcImageSize,
                                     float smoke_thres, std::vector<std::vector<cv::Point>>& contours, bool& isSmoke,
                                     int& binary_pixels, std::vector<std::vector<cv::Point>>& contours_all) {
  isSmoke = false;
  if (feat.empty()) return;
  cv::Mat img;
  feat.convertTo(img, CV_8UC3);

  cv::Point* root_points = new cv::Point[pts.size()];
  for (size_t i = 0; i < pts.size(); i++) {
    root_points[i].x = pts[i].x * 256 / (float)srcImageSize.width;
    root_points[i].y = pts[i].y * 256 / (float)srcImageSize.height;
  }

  const cv::Point* ppt[1] = {root_points};
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
  std::vector<cv::Vec4i> hierarchy;  //
  std::vector<std::vector<cv::Point>> contours_tmp;
  cv::findContours(showimg, contours_tmp, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  int scale_x = (float)srcImageSize.width / (float)showimg.cols;
  int scale_y = (float)srcImageSize.height / (float)showimg.rows;
  // std::vector<std::vector<cv::Point>> contours_filter ;

  for (size_t i = 0; i < contours_tmp.size(); i++) {
    // filter small contours
    double area = cv::contourArea(contours_tmp[i]);
    cv::Rect rect1 = cv::boundingRect(cv::Mat(contours_tmp[i]));
    float tmp = (float)rect1.height / (float)rect1.width;
    float hwratio = tmp > 1 ? tmp : 1 / (float)tmp;
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
        conts_final.push_back(cv::Point(contours_tmp[i][j].x * scale_x + 3, contours_tmp[i][j].y * scale_y + 3));
      }
      contours_all.push_back(conts_final);
    }

  } else {
    isSmoke = false;
    contours.clear();
  }
}
