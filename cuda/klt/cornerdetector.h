#pragma once

#include <stdint.h>
#include <vector>
#include "featurepool.h"


#include "klt/internal/texture.h"

#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/param/parametrized.h>

namespace klt
{




class HarrisDetector:public cvl::Parametrized
{
public:

 struct Candidate
 {
 using Index=uint16_t;
 float quality;
 Index row, col;

 Candidate(Index row, Index col, float quality):row(row),col(col),quality(quality){}

 inline bool operator<( const Candidate& fr_other ) const
 { return quality > fr_other.quality; }

 };

 HarrisDetector( std::string name="Harris Corner Detector");




 // ordered by decreasing quality
 std::vector<cvl::Vector2f>
 detect( const Texture< float, true>& dx,
 const Texture< float, true>& dy,
 const std::vector<cvl::Vector2f>& locked_keypoints);


private:


 void create_buffers(int rows, int cols);
 void compute_mask(const std::vector<cvl::Vector2f>& kps);
 std::vector<cvl::Vector2f>
 candidates(
 const Texture< float, false >& corner_scores_host,
 float min_quality);
 Texture< float, false > locked_keypoints_host;
 Texture< float, true> locked_keypoints_device;
 Texture< float, true> corner_scores;
 Texture< float, true> corner_scores_extra;
 Texture< float, false > corner_scores_host;

 /// Mask buffers
 Texture< float, true> mask_device;
 Texture< float, false> mask_host;
 Texture< float, true> masked_corner_score;



mlib::NamedTimerPack ntp;


 float min_score();
 int radius();
private:
 cvl::RealParameter* min_score_=nullptr;
 cvl::IntParameter* radius_=nullptr;





};


}
