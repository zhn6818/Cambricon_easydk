/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
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
#include <glog/logging.h>
#include <algorithm>  // sort
#include <cstring>    // memset
#include <list>
#include <string>
#include <utility>
#include <vector>

#include "cnpostproc.h"

using std::pair;
using std::vector;
using std::to_string;

namespace edk {

#define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

void CnPostproc::set_threshold(const float threshold) { threshold_ = threshold; }

std::vector<std::vector<DetectObject>> CnPostproc::Execute(const vector<pair<float*, uint64_t>>& net_outputs,int batch) {
  // std::cout << "innnn:"<<std::endl;
  return Postproc(net_outputs,batch);
}

std::vector<std::vector<DetectObject>> ClassificationPostproc::Postproc(const vector<pair<float*, uint64_t>>& net_outputs,int batch) {
  if (net_outputs.size() != 1) {
    LOG(WARNING) << "Classification neuron network only has one output but get "
                    + to_string(net_outputs.size());
  }
std::vector<std::vector<DetectObject>>  results;
  float* data = net_outputs[0].first;
  uint64_t len = net_outputs[0].second;

  for(int j=0;j<batch;++j){
    std::list<DetectObject> objs;
    for (decltype(len) i = 0; i < len; ++i) {


      //if (data[i] < threshold_) continue;
      DetectObject obj;
      memset(&obj.bbox, 0, sizeof(BoundingBox));
      obj.label = i;
      
      obj.score = data[i];
      objs.emplace_back(std::move(obj));
    }
    objs.sort([](const DetectObject& a, const DetectObject& b) { return a.score > b.score; });
    results.push_back(std::vector<DetectObject>(objs.begin(), objs.end()));
    data +=len;
  }
  return results;
}


namespace detail {
template <typename dtype>
struct Clip {
  Clip(dtype _down, dtype _up) : down(_down), up(_up) {}
  inline dtype operator()(dtype val) {
    return std::min(up, std::max(down, val));
  }
  dtype down;
  dtype up;
};
}  // namespace detail

detail::Clip<float> Clip0_1_float(0, 1);

vector<vector<DetectObject>> Yolov3Postproc::Postproc(const vector<pair<float*, uint64_t>>& net_outputs,int batch) {
  vector<vector<DetectObject>> results;
  
  float* data = net_outputs[0].first;
  uint64_t len = net_outputs[0].second;
  constexpr int box_step = 7;
  vector<DetectObject> objs;
  // std::cout<<"len:"<<len<<std::endl;
  // std::cout << "len:"<<len<<std::endl;
  // std::cout << "batch:"<<batch<<std::endl;
  for(int i=0;i<batch;i++)
  {
    objs.clear();
    const int box_num = static_cast<int>(data[0]);
    // std::cout<<"box_num:"<<box_num<<std::endl;
    CHECK_LE(static_cast<uint64_t>(64 + box_num * box_step), len);

    for (int bi = 0; bi < box_num; ++bi) {
      DetectObject obj;
      obj.label = static_cast<int>(data[64 + bi * box_step + 1]);
      obj.score = data[64 + bi * box_step + 2];
      // if (obj.label == 0) continue;
      if (threshold_ > 0 && obj.score < threshold_) continue;
      obj.bbox.x = Clip0_1_float(data[64 + bi * box_step + 3]);
      obj.bbox.y = Clip0_1_float(data[64 + bi * box_step + 4]);
      obj.bbox.width = Clip0_1_float(data[64 + bi * box_step + 5]) - obj.bbox.x;
      obj.bbox.height = Clip0_1_float(data[64 + bi * box_step + 6]) - obj.bbox.y;

      obj.bbox.x = (obj.bbox.x - padl_ratio_) / (1 - padl_ratio_ - padr_ratio_);
      obj.bbox.y = (obj.bbox.y - padt_ratio_) / (1 - padb_ratio_ - padt_ratio_);
      obj.bbox.width /= (1 - padl_ratio_ - padr_ratio_);
      obj.bbox.height /= (1 - padb_ratio_ - padt_ratio_);

      obj.track_id = -1;
      if (obj.bbox.width <= 0) continue;
      if (obj.bbox.height <= 0) continue;
      objs.push_back(obj);
    }
    data +=len/batch;
    results.push_back(objs);
  }
  
  return results;
}

vector<vector<DetectObject>> Yolov4Postproc::Postproc(const vector<pair<float*, uint64_t>>& net_outputs,int batch) {
  vector<vector<DetectObject>> results;
  
  float* data = net_outputs[0].first;
  uint64_t len = net_outputs[0].second;
  constexpr int box_step = 7;
  vector<DetectObject> objs;
  // std::cout<<"len:"<<len<<std::endl;
  // std::cout << "len:"<<len<<std::endl;
  // std::cout << "batch:"<<batch<<std::endl;
  for(int i=0;i<batch;i++)
  {
    objs.clear();
    const int box_num = static_cast<int>(data[0]);
    //std::cout<<"box_num:"<<box_num<<std::endl;
    CHECK_LE(static_cast<uint64_t>(64 + box_num * box_step), len);

    for (int bi = 0; bi < box_num; ++bi) {
      DetectObject obj;
      obj.label = static_cast<int>(data[64 + bi * box_step + 1]);
      obj.score = data[64 + bi * box_step + 2];
      // if (obj.label == 0) continue;
      if (threshold_ > 0 && obj.score < threshold_) continue;
      obj.bbox.x = (data[64 + bi * box_step + 3]);
      obj.bbox.y = (data[64 + bi * box_step + 4]);
      obj.bbox.width = (data[64 + bi * box_step + 5]) - obj.bbox.x;
      obj.bbox.height = (data[64 + bi * box_step + 6]) - obj.bbox.y;

      obj.bbox.x = (obj.bbox.x - padl_ratio_) / (1 - padl_ratio_ - padr_ratio_);
      obj.bbox.y = (obj.bbox.y - padt_ratio_) / (1 - padb_ratio_ - padt_ratio_);
      obj.bbox.width /= (1 - padl_ratio_ - padr_ratio_);
      obj.bbox.height /= (1 - padb_ratio_ - padt_ratio_);

      obj.track_id = -1;
      if (obj.bbox.width <= 0) continue;
      if (obj.bbox.height <= 0) continue;
      objs.push_back(obj);
    }
    data +=len/batch;
    results.push_back(objs);
  }
  
  return results;
}
}  // namespace edk
