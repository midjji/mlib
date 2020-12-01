#pragma once

/* ********************************* FILE ************************************/
/** \file    dataset.h
 *
 * \brief    This header contains the wrapper for the kitti mots dataset
 *
 * \remark
 * - c++11
 *
 * \todo
 *
 *
 *
 * \author   Mikael Persson
 * \date     2019-01-01
 * \note GPL licence
 *
 ******************************************************************************/

#include <kitti/mots/sample.h>
#include <kitti/mots/calibration.h>
namespace cvl{

/**
 * @brief The KittiMotsDataset class
 * has rgb: left,right, and calib which varies with sequence.
 *
 */
class KittiMotsDataset
{

public:
    KittiMotsStereoCalibration calibration(bool training, int sequence){
        return KittiMotsCalibration::get_color_stereo(path,training,sequence);
    }
    KittiMotsDataset(std::string dataset_path);
    //std::vector<int> sequence_images={        232,        39,        313,        296,        269        799,        389,        802,        293,        372,        77,        339        105        375        169,        144,        338,        1058,        836    }
int sequences(bool train);
void write_sample_paths(std::string path);
    std::shared_ptr<KittiMotsSample> get_sample(
            bool training,
            uint sequence,
            uint frameid);
    int samples(bool training, uint sequence);
    std::shared_ptr<KittiMotsSample>  next(){
        if(index<50) index=150;
        return get_sample(true,5,index++);}

    int rows(bool training, int seq);
    int cols(bool training, int seq);

private:
    int index=0;
    std::string path;
    int test_sequences=28;
    int train_sequences=21;
    // rows,cols per sequence...
    Vector2<uint> max_train_row_col();
    std::vector<Vector2<uint>>  training_row_col{{375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {370, 1224}, {370, 1224}, {370, 1224}, {370, 1224}, {374, 1238}, {374, 1238}, {376, 1241} };
    std::vector<Vector2<uint>>  testing_row_col{{375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {375, 1242}, {370, 1224}, {370, 1224}, {370, 1224}, {370, 1224}, {370, 1224}, {370, 1224}, {370, 1224}, {370, 1224}, {370, 1224}, {370, 1224}, {370, 1226}};
    std::vector<uint> train_samples{154, 447, 233, 144, 314, 297, 270, 800, 390, 803, 294, 373, 78, 340, 106, 376, 209, 145, 339, 1059, 837 };
    std::vector<uint> test_samples {465, 147, 243, 257, 421, 809, 114, 215, 165, 349, 1176, 774, 694, 152, 850, 701, 510, 305, 180, 404, 173, 203, 436, 430, 316, 176, 170, 85 };
    void check_count();
    void sanity_check_calibrations();


};



} // end namespace daimler
