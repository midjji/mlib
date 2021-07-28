#include <mlib/datasets/kitti/odometry/distlsh.h>
#include <mlib/utils/mlibtime.h>
#include <mlib/utils/mlog/log.h>
namespace cvl{
namespace kitti{

/**
 * @brief getDistanceTraveled
 * @param ps
 * @return a vector listing how far the car has travelled at time t
 *
 *  distance[i]= sum(|translation(t) -translation(t-1)|)
 *
 * Assumes transform:
 *
 * X_w=P_wc*X_c ie the inverse of the usual
 *
 * Note this function is used on the gt poses to create the sampling for the eval
 */

std::vector<double> getDistanceTraveled(const std::vector<cvl::PoseD>& ps){
    std::vector<double> distances;distances.reserve(ps.size()+1);
    distances.push_back(0);
    for(uint t=1;t<ps.size();++t){
        //these have to be in the kitti structure!
        double relative=(ps[t].getT()-ps[t-1].getT()).length();
        distances.push_back(distances.at(t-1)+relative);
    }
    return distances;
}

DistLsh::operator bool() const{return !dists.empty();}
namespace  {
mlib::Timer timer("distlsh");
}
DistLsh::DistLsh(const std::vector<cvl::PoseD>& ps)
{
    mlog()<<"begin\n";
    timer.tic();
    // potentially heavy constructor...
    dists=getDistanceTraveled(ps);

    // assume minimum is 0
    // assume 0-lots per meter
    // actually it might be faster to use less dense buckets well typically there will be zero to three per bucket as is, should be pretty fast
    distmap.resize((int)(dists.back()+2));

    for(std::vector<std::pair<uint,double>>& v:distmap)
        v.reserve(64);
    for(uint i=0;i<dists.size();++i){
        distmap.at((uint)dists[i]).push_back(std::make_pair(i,dists[i]));
    }
    timer.toc();
    mlog()<<timer<<"\n";
}

int DistLsh::getIndexPlusDist(uint index, double dist) const{

    if(index>dists.size()) return -2;
    double a=dists.at(index);
    double b=a+dist;
    int bindex=int(std::floor(b)-1);
    if(bindex<0) bindex=0;
    // for each possible index position test in order returning the first match or if all fail return -1
    for(uint m=bindex;m<distmap.size();++m){ // matches behaviour of original despite the problem it implies if there are long jumps
        for(const std::pair<uint,double>& distv : distmap.at(m))
            if(distv.second>b)
                return distv.first;
    }
    return -1;
}

}
}
