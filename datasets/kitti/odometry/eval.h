#pragma once
#include <kitti/odometry/result.h>

namespace cvl{
namespace kitti{


/**
 * @brief The Evaluator class provides the evaluation output for the kitti benchmark
 *
 *
 * \todo
 * - support for extended gt in evaluation
 *
 */
class Evaluator{

public:
    /**
     * @brief Evaluator basic constructor
     */
    Evaluator(std::string basepath      /** @param basepath     path to the kitti dataset */,
              std::vector<std::string> estimatepath  /** @param estimatepath path to the estimation output directories */,
              std::vector<std::string> names  /** estimate names */,
              std::string outputpath    /** @param outputpath   where to store teh resulting files */);



    /// perform the evaluation and save the results
    void evaluate();


private:
    /// costly internal initialization
    void init();
    bool inited=false;

    KittiDataset kd;
    std::vector<Result> results;
    std::vector<std::string> estimatepaths,names;
    std::string output_path="./";

};
}// end kitti namespace
}// end namespace cvl
