#include <mlib/utils/nd_bucket_map.h>
#include <cmath>
#include <algorithm>

namespace cvl{
Fast2DQuery::Fast2DQuery(Vector2d minv,
                         Vector2d maxv,
                         Vector2i buckets)
{
    delta[0]=std::ceil((maxv[0]-minv[0])/double(buckets[0]));
    delta[1]=std::ceil((maxv[1]-minv[1])/double(buckets[1]));
    griddata.resize(buckets[0]*buckets[1],std::vector<Data>()); // essentially holds a pointer in each element
    for(auto& v:griddata) v.reserve(64);
    grid=MatrixAdapter<std::vector<Data>>(&griddata[0],buckets[0],buckets[1]);
}
Vector2i Fast2DQuery::position(Vector2d y){
    y=y-minv;
    y[0]/=delta[0];
    y[1]/=delta[1];
    y.cap(Vector2d(0,0),grid.dimensions());
    Vector2i index(y[0],y[1]); // rounds down...
    return index;
}




double minimum_distance(Vector2d y, Vector2i index, Vector2d delta){
    double a=(delta.pointMultiply(index)             - y).norm2();
    double b=(delta.pointMultiply(index+Vector2i(1,1))- y).norm2();
    double c=(delta.pointMultiply(index+Vector2i(0,1))- y).norm2();
    double d=(delta.pointMultiply(index+Vector2i(1,0))- y).norm2();
    return std::min(std::min(a,b),std::min(c,d) );
}


int Fast2DQuery::find(
        Vector2d y,
        double max_radius)
{

    Vector2i p=position(y);
    Vector2i low=p-Vector2i(1,1);       low.cap(Vector2d(0,0),grid.dimensions());
    Vector2i high=p+Vector2i(1,1);       high.cap(Vector2d(0,0),grid.dimensions());
    Data best; best.index=-1;
    double best_dist=std::numeric_limits<double>::max();
    for(int r=low[0];r<high[0];++r)
        for(int c=low[1];c<high[1];++c)
            for(auto& data:grid(Vector2i(r,c))){
                double dist=(data.x - y).norm2();
                if(best_dist< dist) continue;
                if(dist>max_radius*max_radius) continue;
                best_dist=dist;
                best=data;
            }
    return best.index;
}

void Fast2DQuery::add(Fast2DQuery::Data data){
    auto y=position(data.x);
    grid(y[0],y[1]).push_back(data);
}
}
