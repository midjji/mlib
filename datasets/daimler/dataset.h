#pragma once

/* ********************************* FILE ************************************/
/** \file    daimler_dataset.h
 *
 * \brief    This header contains the wrapper for the sequences I got from daimler...
 *
 * \remark
 * - c++11
 *
 * \todo
 * - add more sequences?
 * - fix the issue with the missing label samples, this also misses a few images since size is off...
 *
 *
 *
 * \author   Mikael Persson
 * \date     2019-01-01
 * \note GPL licence
 *
 ******************************************************************************/

#include <mlib/datasets/daimler/sample.h>
#include <mlib/datasets/daimler/database.h>
namespace cvl{

/**
 * @brief The DaimlerDataset class
 * Convenient daimlerdata,
 * it has
 * left, right, disparity, calib, metadata, semantic labels, instance labels an<T> bb annotations
 *
 * top path is xx/08 or xx/06 there are two of them. include 08 in the dataset path
 *
 */
class DaimlerDataset
{




public:

    using sample_type=std::shared_ptr<DaimlerSample>;

    DaimlerDataset(std::string dataset_path, std::string gt_path="");
    std::string path;
    mtable::gt_db_type gt_storage;
    std::shared_ptr<DaimlerSample> get_sample(uint index);
    int samples() const;
    double fps() const;

    std::vector<PoseD> gt_poses; // interface..
    std::vector<PoseD> gt_vehicle_poses();

private:
    uint total_samples=0;
    cv::Mat1b get_cars(cv::Mat1b labels);
    bool read_images(uint sampleindex,
                     cv::Mat1w& left,
                     cv::Mat1w& right,
                     cv::Mat1b& labels,
                     cv::Mat1f& disparity);



    bool read_right=true;
};

} // end namespace daimler
