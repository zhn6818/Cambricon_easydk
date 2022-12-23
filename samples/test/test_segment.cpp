
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
DEFINE_string(data_path, "/data/ld/project/Model_To_Cambricon-master/cvt_model_data/segmention/img/1_2_37.jpg",
              "video path");
DEFINE_string(model_path, "/data/ld/project/instanceseg/model_yanwu/cambricion/best_model_b4c4.cambricon",
              "infer offline model path");
// DEFINE_string(model_path, "/data1/zhn/instanceseg/model_yanwu/cambricion/best_model_b1c4.cambricon",
//               "infer offline model path");

DEFINE_string(func_name, "subnet0", "model function name");
DEFINE_int32(wait_time, 0, "time of one test case");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::cout << "instanceseg" << std::endl;
  CHECK_NE(FLAGS_data_path.size(), 0u);
  CHECK_NE(FLAGS_model_path.size(), 0u);
  CHECK_NE(FLAGS_func_name.size(), 0u);

  CHECK_GE(FLAGS_wait_time, 0);
  CHECK_GE(FLAGS_repeat_time, 0);
  std::cout << FLAGS_data_path << std::endl;
  std::cout << FLAGS_model_path << std::endl;
  std::cout << FLAGS_func_name << std::endl;
  std::cout << FLAGS_wait_time << std::endl;
  std::cout << FLAGS_repeat_time << std::endl;
  std::cout << std::endl;

  std::vector<cv::Point> pts;
  std::shared_ptr<Segment> SegmentPtr = std::make_shared<Segment>(FLAGS_model_path, FLAGS_func_name, 0);

  cv::Mat img = cv::imread("/data1/zhn/instanceseg/model_yanwu/img/2_fin-hk-2019-05-24-16h03m46s289.jpg");
  cv::Mat img2 = img.clone();
  std::vector<cv::Mat> vBatch;
  vBatch.push_back(img2);
  vBatch.push_back(img2);
  vBatch.push_back(img2);
  vBatch.push_back(img2);
  cv::Mat feat;
  SegmentPtr->getfeat(vBatch, feat);
  std::cout << "getfeat: " << std::endl;
  std::cout << "batch size : " << SegmentPtr->GetBatch() << std::endl;
  cv::imwrite("/data/ld/project/git/samples/test/segment_0.jpg", feat);
  //   cv::imwrite("/data/ld/project/git/samples/test/segment_1.jpg", feats[1]);
  //   cv::imwrite("/data/ld/project/git/samples/test/segment_2.jpg", feats[2]);
  //   cv::imwrite("/data/ld/project/git/samples/test/segment_3.jpg", feats[3]);
  //   int index = 0;
  //   for (auto& feat : feats) {
  //     cv::Mat tmpImg = img2.clone();
  //     bool isSmoke = false;
  //     std::vector<std::vector<cv::Point>> contours;
  //     int binary_pixels = 0;
  //     pts = {cv::Point(0, 0), cv::Point(img.cols - 1, 0), cv::Point(img.cols - 1, img.rows - 1),
  //            cv::Point(0, img.rows - 1), cv::Point(0, 0)};
  //     float smoke_thres = 200;
  //     std::vector<std::vector<cv::Point>> contours_all;
  //     SegmentPtr->processFeat_test(feat, pts, cv::Size(img.cols, img.rows), smoke_thres, contours, isSmoke,
  //     binary_pixels,
  //                                  contours_all);

  //     std::cout << "isSmoke: " << isSmoke << std::endl;

  //     std::cout << "coontours size: " << contours.size() << std::endl;

  //     for (size_t i = 0; i < contours.size(); i++) {
  //       cv::polylines(tmpImg, contours[i], false, cv::Scalar(0, 255, 0), 2);
  //     }
  //     for (size_t i = 0; i < contours_all.size(); i++) {
  //       cv::polylines(tmpImg, contours_all[i], false, cv::Scalar(0, 0, 255), 2);
  //     }
  //     cv::imwrite("/data/ld/project/git/samples/test/segment_res_" + std::to_string(index) + ".jpg", tmpImg);
  //     index += 1;
  //   }

  return 0;
}