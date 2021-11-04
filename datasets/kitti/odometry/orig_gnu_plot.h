#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <mlib/utils/cvl/pose.h>

namespace cvl{
namespace kitti{
struct KittiError;

/**
 * @brief plot_errors creates the average error .eps files which mirror the ones which will be present in the benchmark
 * @param es            the kitti errors in question
 * @param output_path   the directory to write too
 * @param name          name of the sequence
 * @param lengths       the distances where measurements should be taken
 * plots the kitti errors using gnuplot
 */
void plot_errors(std::vector<KittiError>& es, std::string output_path, std::string name,std::vector<double> lengths);


void plot_sequence(std::vector<cvl::PoseD> gt,std::vector<cvl::PoseD> res,std::string label,
                   std::string output_path,
                   std::string name);
void plot_sequence(std::vector<cvl::PoseD> gt,std::vector<std::vector<cvl::PoseD>> res,std::vector<std::string> labels,
                   std::string output_path,
                   std::string name);

}// end kitti namespace
}// end namespace cvl
