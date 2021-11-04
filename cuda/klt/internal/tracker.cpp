#include "klt/tracker.h"
#include "klt/featurepool.h"
#include "klt/internal/pyramid.h"
#include "klt/internal/pyramid_utils.h"
#include "klt/internal/scharr.h"

#include "klt/internal/klt_kernels.h"
#include "klt/internal/texture.h"
#include "klt/internal/pyramid.h"

#include "klt/cornerdetector.h"

#include <mlib/utils/mlog/log.h>

namespace klt
{

struct TrackerSample{
    Pyramid< Texture< float > > image;
    Pyramid< Texture< float > > dx;
    Pyramid< Texture< float > > dy;
};

void set_features(const FeaturePool& pool,
                  Texture< SCUDAKLTFeature_t,false >& feature_pool_host)
{
    feature_pool_host.resize_array(       pool.size());
    for ( int i=0; i<pool.size(); ++i)
    {
        const auto& feature=pool[i];
        auto& cuda_feature=feature_pool_host(0,i);
        cuda_feature=feature.klt_feature(true);
    }
}



void update_pool(FeaturePool& pool,
                 const Texture< SCUDAKLTFeature_t,false >& feature_pool_host,
                 double max_residual )
{

    for ( int i=0; i<pool.size(); ++i)
    {

        auto& feature=pool[i];
        const auto& cuda_feature=feature_pool_host(0,i);
        if(!feature.tracked_or_found()) continue;
        feature.update(cuda_feature, max_residual);
    }
}

FeaturePool&  Tracker::getFeaturePool()
{
    return pool;
}
TrackerSample& Tracker::current()  { if(Acurrent) return *A; return *B;}
TrackerSample& Tracker::previous() { if(Acurrent) return *B; return *A;}
Texture< float,false>& Tracker::target_image_host(){return *target_image_host_;}
Texture< float,true>&  Tracker::target_image_device(){return *target_image_device_;}
Texture< SCUDAKLTFeature_t, true >& Tracker::feature_pool_device(){return *feature_pool_device_;}
Texture< SCUDAKLTFeature_t, false>& Tracker::feature_pool_host(){return *feature_pool_host_;}
void Tracker::track(const cv::Mat1f& image)
{
    mhere();

    // reshuffle the data so the stride matches
    // could possibly be avoided sometimes, but its not a serious bottleneck
    auto& tih=target_image_host();
    tih.resize_rc(image.rows,image.cols); // note be explicit here, since opencv stride is different
    mhere();
    for(int r=0;r<image.rows;++r)
        for(int c=0;c<image.cols;++c)
            tih(r,c)=image(r,c);
    mhere();
    // reuses the buffert
    target_image_device()=tih;
    mhere();
    trackit();
}

namespace{
void cap(int& a, int b, int c){
    if(a<b) a=b;
    if(a>c) a=c;
}}

Tracker::Tracker(  const std::string&  name )
    : Parametrized(name," klt tracker remake")
{



    harris=new HarrisDetector();
    target_image_host_    = new Texture< float,false>();
    target_image_device_  = new Texture< float,true>();
    A                     = new TrackerSample();
    B                     = new TrackerSample();
    feature_pool_device_  = new Texture< SCUDAKLTFeature_t,true>();
    feature_pool_host_    = new Texture< SCUDAKLTFeature_t,false>();


    add("corner_detector", harris->params());

    // tracker subtype,

    tracker_subtype=pint(0, "tracker subtype", "Tracker", "0 standard, 1 mean, 2 1D",0,3);

    pyramid_levels=pint(4,"number of pyramid levels","Tracker", "0 means only original image?", 0);
    pool_size=pint(10000, "Maximum number of tracks", "Tracker","",0);

    half_window=pint(3, "half_window->value()", "Optimizer","",1,11);
    num_iterations=pint(40, "max iterations", "Optimizer","per pyramid level",3,100);
    min_displacement=preal(0.1, "min delta", "Optimizer"," for the optimzer to be considered to have converged",0.01,0.5);
    max_displacement=preal(1, "max delta", "Optimizer"," longest allowed step ",0.1,10);
    max_residual=preal(-1, "max ssd error ", "Optimizer"," <0 for any ");
    mlog()<<display()<<"\n";
}
Tracker::~Tracker(){
    delete harris;
    delete target_image_host_;
    delete target_image_device_;
    delete A;
    delete B;
    delete feature_pool_device_;
    delete feature_pool_host_;
}












void
Tracker::reset()
{
    mlog()<<"resetting\n";

    mhere();
    pool.reset();
    previous_valid=false;


}
void Tracker::klt(const TrackerSample& prev, const TrackerSample& curr, int stream)
{
    switch ( tracker_subtype->value() )
    {
    case 0:
        cudaKLT( prev.image,
                 curr.image,
                 prev.dx,
                 prev.dy,
                 feature_pool_device(),
                 half_window->value(),
                 num_iterations->value(),
                 min_displacement->value(),
                 max_displacement->value(),
                 stream );
        return;
    case 1:
        cudaKLT_ZMSSD( prev.image,
                       curr.image,
                       prev.dx,
                       prev.dy,
                       feature_pool_device(),
                       half_window->value(),
                       num_iterations->value(),
                       min_displacement->value(),
                       max_displacement->value(),
                       stream  );
        return;
}
}




void
Tracker::trackit(  )
{
    // update all parameters from the gui, they wont change untill the next time this function is called.
    update_all();
    pool.resize(pool_size->value());
    mhere();
    TrackerSample& curr=current();
    TrackerSample& prev=previous();


    // we have two streams, strictly speaking, so lots of speedup possible...
    // leave it for now
    mhere();
    pyramid_gauss3x3_gauss5x5( target_image_device(), curr.image, pyramid_levels->value() );
    mhere();
    // Set all previous lost features to undefined state.
    pool.clear_lost();
    mhere();
    if(previous_valid) // track
    {

        //"Track - Upload features" );
        set_features(pool, feature_pool_host() );
        feature_pool_device()       = feature_pool_host();
        ////  "Track - Prepare Gradient Pyramid" );
        klt( prev, curr);
        mhere();




        mhere();

        // "Track - Download features" );
        feature_pool_host()       = feature_pool_device();
        mhere();
        update_pool(pool, feature_pool_host(), max_residual->value() );
        mhere();
    }
    mhere();

    scharr_pyramid(curr.image, curr.dx, curr.dy);


    mhere();

    auto kps=harris->detect( curr.dx.getImage(0),
                            curr.dy.getImage(0),
                            pool.valid_kps());
    pool.assign_new(kps,false);
    mhere();

    Acurrent=!Acurrent;
    previous_valid=true;
}










}
