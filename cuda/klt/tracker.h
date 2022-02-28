#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <string>
#include <opencv2/core/mat.hpp>
#include <mlib/param/parametrized.h>
#include "mlib/cuda/klt/featurepool.h"



template<class T, bool device=true> class Texture;
namespace klt
{


class TrackerSample;
class HarrisDetector;


class Tracker : public cvl::Parametrized
{


    FeaturePool           pool;


public:

    Tracker( const std::string&  name =
            std::string( "CUDA KLT Tracker" ) );

    Tracker( Tracker& fr_other )=delete;
    Tracker& operator=( const Tracker& )=delete;
    ~Tracker();


    /**
     * @brief track
     * @param image
     *
     * tracks into the provided image, and updates which image we are tracking from and computes harris corners etc, updates feature pool etc...
     */
    void track(const cv::Mat1f& image, bool replace_previous=true, bool detect_new=true);
    FeaturePool&  getFeaturePool();



private:

    void reset();

    void trackit();

    /// Corner detector
    HarrisDetector* harris;
    Texture< float,false>* target_image_host_;
    Texture< float,true>*  target_image_device_;
    Texture< float,false>& target_image_host();
    Texture< float,true>&  target_image_device();


    bool previous_valid=false;
    bool Acurrent=true;

    TrackerSample& current();
    TrackerSample& previous();
    TrackerSample* A;
    TrackerSample* B;

    Texture< SCUDAKLTFeature_t, true >* feature_pool_device_;
    Texture< SCUDAKLTFeature_t, false>* feature_pool_host_  ;

    Texture< SCUDAKLTFeature_t, true >& feature_pool_device();
    Texture< SCUDAKLTFeature_t, false>& feature_pool_host();



    // tracker subtype,
    cvl::IntParameter*     tracker_subtype;
    /// Number of pyramid levels including level 0 for original image.
    cvl::IntParameter*     pyramid_levels;
    cvl::IntParameter*     pool_size;
    cvl::IntParameter*     half_window;
    cvl::IntParameter*     num_iterations; // per pyramid levels

    /// convergence displacement
    cvl::RealParameter*        min_displacement;
    cvl::RealParameter*        max_displacement;


    /// Maximum average squared difference of the same feature in old and current image.
    cvl::RealParameter*       max_residual;

    void klt(const TrackerSample& prev,
             const TrackerSample& curr,
             int stream=0);


};






}




