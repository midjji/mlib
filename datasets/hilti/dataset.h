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

#include <mlib/datasets/hilti/sequence.h>
#include <mlib/utils/cvl/bimap.h>
namespace cvl {


namespace hilti {



struct Dataset {


    std::map<int, std::string > num2sequence{
            {0, "Construction_Site_1"}
    //{8, "uzh_tracking_area_run2"},
    //{1, "Basement_1"},
    //{2, "Basement_4"},
    //{3, "Campus_2"},
    //{4, "Construction_Site_2"},
    //{5, "Office_Mitte_1"},
    //{6, "Basement_3"},
    //{7, "Campus_1"},

    //{9, "IC_Office_1"},
    //{10, "LAB_Survey_2"},
    //{11, "Parking_1"}
    };

    std::map<int, std::shared_ptr<Sequence>> seqs;
    std::shared_ptr<Sequence> sequence(int index) const;

    Dataset(std::string dataset_path="/storage/datasets/hilti/preprocessed/");
};


const Dataset& dataset(std::string path="/storage/datasets/hilti/preprocessed/");



}
}

