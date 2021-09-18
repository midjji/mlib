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
namespace cvl {


namespace hilti {



struct Dataset {



    std::map<std::string, bool> sequence_names_=
    {
        {"rpg_drone_testing_arena",true},
       // {"ic_office",false},
       // {"office_mitte",false},
       // {"parking_deck",false},
       // {"basement",true},
       // {"basement_3",false},
       // {"basement_4",true},
       // {"lab",true},
       // {"construction_site_outdoor_1",false},
       // {"construction_site_outdoor_2",true},
       // {"campus_1",false},
       // {"campus_2",true}
    };


    std::vector<Sequence> seqs;

    Dataset(std::string dataset_path="/storage/datasets/hilti/");
};


}
}

