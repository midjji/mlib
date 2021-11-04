#include <cstdio>
#include "klt/featurepool.h"
#include <mlib/opencv_util/draw_arrow.h>
#include <mlib/opencv_util/cv.h>
namespace klt {


std::ostream& operator<<(std::ostream &os, SFeature_t t){
    return os<<t.str();
}

std::string FeaturePool::str() const{
    std::stringstream ss;
    ss<<"feature pool: "<<pool.size();
    return ss.str();
}
std::vector<cvl::Vector2f> FeaturePool::valid_kps() const{
    std::vector<cvl::Vector2f> kps;kps.reserve(pool.size());
    for(const auto& f:pool){
        if(f.tracked_or_found())
            kps.emplace_back(f.row(),f.col());
    }
    return kps;
}
void FeaturePool::assign_new(
        const std::vector<cvl::Vector2f>& kps,
        bool grow_if_required)
{
    if(grow_if_required && pool.capacity()<pool.size()+kps.size())
        pool.reserve(pool.size()+kps.size());
    int i=0;
    for(const auto& f:kps)
    {
        for(;i<pool.size();++i)
        {
            auto& pf=pool[i];
            if(!pf.tracked_or_found()){
                pf=SFeature_t(f[0],f[1]);
                break;
            }
        }
        if(i<pool.size()) continue;
        if(!grow_if_required) break;
        pool.push_back(SFeature_t(f[0],f[1]));
    }
}


FeaturePool::FeaturePool(int initial_size, int reserve  ){
    pool.reserve(reserve);
    pool.resize(initial_size);
}
SFeature_t* FeaturePool::begin(){
    return &pool[0];
}
const SFeature_t* FeaturePool::begin() const{
    return &pool[0];
}
const SFeature_t* FeaturePool::end() const{
    // size +1
    return &pool[0] + size();
}

void FeaturePool::resize( int new_size ){    pool.resize(new_size);}
int  FeaturePool::size() const   {       return pool.size();    }
void FeaturePool::reserve(int size)  {       pool.reserve(size);    }
SFeature_t& FeaturePool::operator[](int index) {  return pool[index];}
const SFeature_t& FeaturePool::operator[](int index) const {  return pool[index];}

void
FeaturePool::reset()
{
    int size=pool.size();
    pool.resize(0);
    pool.resize(size);
}
void FeaturePool::clear_lost(){
    for(auto& p:pool){
        if(!p.is_lost_or_undefined()) continue;
        p=SFeature_t();
    }
}


void
FeaturePool::print() const
{
    // Get number of features of each state
    int free = 0;
    int found     = 0;
    int tracked   = 0;
    int lost      = 0;

    for (const auto& f:pool)
    {
        if(f.free()) free++;
        if(f.found()) found++;
        if(f.tracked()) tracked++;
        if(f.lost()) lost++;
    }


    printf( "------------------------\n"
            "|    Feature pool      |\n"
            "------------------------\n"
            " Size:      %d\n"
            " Free: %d\n"
            " Found:     %d\n"
            " Tracked:   %d\n"
            " Lost:      %d\n"
            "------------------------\n",
            int(pool.size()),
            free,
            found,
            tracked,
            lost );

    // Display all features
    int i=0;
    for (const auto& f:pool)

std::cout<<f<<std::endl;
    printf( "------------------------\n" );
}
cv::Mat3b draw_feature_pool(klt::FeaturePool& pool,
                            cv::Mat3b rgb){
    //cout<<"drawRawTracks"<<endl;



    for(auto f:pool){
        if(f.found()){
            mlib::draw_circle(rgb,f.rc<cvl::Vector2d>(),mlib::Color::green());
        }
        if(f.tracked()){
            mlib::drawArrow(rgb,f.rc<cvl::Vector2d>(),f.previous_rc<cvl::Vector2d>(), mlib::Color::blue());
        }
    }
    //  cout<<"drawRawTracks - done"<<endl;
    return rgb;
}
cv::Mat3b draw_feature_pool_prediction(const FeaturePool& pool, cv::Mat3b rgb){

    for(const auto& f:pool)
    {

        if(f.tracked())
            mlib::drawArrow(rgb, f.rc<cvl::Vector2d>(),f.predicted_rc<cvl::Vector2d>(),mlib::Color::blue());
        if(f.found()){
            auto yp=f.predicted_rc<cvl::Vector2d>();
            //auto y=f.rc<Vector2d>();
            if(!yp.in(cvl::Vector2d(1,1), cvl::Vector2d(rgb.rows-1,rgb.cols-1)))  continue;
            mlib::drawArrow(rgb, f.rc<cvl::Vector2d>(),yp,mlib::Color::green());
        }
    }
    return rgb;

}
}
