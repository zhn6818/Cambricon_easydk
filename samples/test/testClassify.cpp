
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
DEFINE_int32(repeat_time, 0, "process repeat time");
DEFINE_string(data_path, "", "video path");
// DEFINE_string(model_path, "/data1/zhn/resnet18_hanwuji/code/resnet18__intx.cambricon", "infer offline model path");
DEFINE_string(model_path, "/data1/zhn/baojie/model_resnet18_intx.cambricon", "infer offline model path");
// DEFINE_string(model_path, "/data1/zhn/baojie/model_resnet18_intx.cambricon", "infer offline model path");

DEFINE_string(func_name, "subnet0", "model function name");
DEFINE_int32(wait_time, 0, "time of one test case");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // check params
  // CHECK_NE(FLAGS_data_path.size(), 0u);
  CHECK_NE(FLAGS_model_path.size(), 0u);
  CHECK_NE(FLAGS_func_name.size(), 0u);

  CHECK_GE(FLAGS_wait_time, 0);
  CHECK_GE(FLAGS_repeat_time, 0);
  std::cout << "FLAGS_func_name:" << FLAGS_func_name << std::endl;
  std::string names_file = "/data1/zhn/baojie/name.txt";
  std::shared_ptr<Classifycation> g_runner =
      std::make_shared<Classifycation>(FLAGS_model_path, FLAGS_func_name, names_file, 0);

  std::cout << "initial success" << std::endl;
  //   std::string img_name1 = "/data1/zhn/resnet18_hanwuji/data/off/103_16001.jpg";
  //   std::string img_name1 = "/data1/zhn/resnet18_hanwuji/data/red/103_450013.jpg";
  //   std::string img_name1 = "/data1/zhn/resnet18_hanwuji/data/neg/50000_light-state2021-06-08-10-20-18_113_3.jpg";

  //   std::string img_name1 = "/data1/zhn/resnet18_hanwuji/data/green/963_green.jpg";
  std::vector<std::string> vecFilename;
  std::string filePath = "/data1/zhn/baojie/data_baojie/file.txt";
  std::ifstream fin(filePath, std::ios::in);
  char line[1024] = {0};
  std::string name = "";
  while (fin.getline(line, sizeof(line))) {
    std::stringstream word(line);
    word >> name;
    // std::cout << "name: " << name << std::endl;
    vecFilename.push_back(name);
  }
  fin.clear();
  fin.close();
  std::cout << "total images: " << vecFilename.size() << std::endl;
  while (1) {
    for (size_t ii = 0; ii < vecFilename.size(); ii++) {
      std::string img_name1 = vecFilename[ii];
      std::cout << "img_name1: " << img_name1 << std ::endl;
      cv::Mat frame1 = cv::imread(img_name1);
      std::vector<cv::Mat> batch_image;
      batch_image.push_back(frame1);
      double t1 = (double)cv::getTickCount();
      std::vector<std::vector<Prediction>> result;
      
      result = g_runner->Classify(batch_image);
      t1 = (double)cv::getTickCount() - t1;
      std::cout << "compute time:    " << t1 * 1000 / cv::getTickFrequency() << "ms " << std::endl;
      for (int g = 0; g < 1; ++g) {
        std::cout << "label:" << result[g][0].first << std::endl;
        std::cout << "confidence:" << result[g][0].second << std::endl;
      }
      cv::waitKey(30);
    }
  }

  return 0;
}