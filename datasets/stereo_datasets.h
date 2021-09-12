#pragma once
#include <mlib/datasets/stereo_dataset.h>
#include <mlib/utils/buffered_dataset_stream.h>
namespace cvl {

std::shared_ptr<StereoSequence>
kitti_sequence(int sequence=0,
               std::string path="/storage/datasets/kitti/odometry/");

std::shared_ptr<StereoSequence>
daimler_sequence(std::string path="/storage/datasets/daimler/2020-04-26/08",
        std::string gt_path="");



cvl::BufferedStream<StereoSequence>
buffered_kitti_sequence(
        int sequence=0,
        int offset=0,
        std::string path="/storage/datasets/kitti/odometry/");

cvl::BufferedStream<StereoSequence>
buffered_daimler_sequence(
        int offset=0,
        std::string path="/storage/datasets/daimler/2020-04-26/08",
        std::string gt_path="");
}

