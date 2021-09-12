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

#include <mlib/datasets/stereo_dataset.h>
#include <mlib/datasets/daimler/sequence.h>

namespace cvl{

struct DaimlerDataset :public StereoDataset{
    std::shared_ptr<DaimlerSequence> seq;
    std::vector<std::shared_ptr<StereoSequence>> sequences() const;
    DaimlerDataset(std::string dataset_path, std::string gt_path="");
};


} // end namespace daimler
