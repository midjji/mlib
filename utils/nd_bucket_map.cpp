#include <mlib/utils/nd_bucket_map.h>
#include <cmath>
#include <algorithm>

namespace cvl{
Fast2DQuery::Fast2DQuery(Vector2d minv,
                         Vector2d maxv, Vector2i buckets):minv(minv),maxv(maxv)
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
    Vector2i indexes(y[0],y[1]); // rounds down...
    return indexes;
}


std::vector<Vector2i> Fast2DQuery::query_locations(Vector2d y, double max_radius){
    // replace with fast generator later...
    Vector2i low=position(y-Vector2d(max_radius,max_radius));
    Vector2i high=position(y-Vector2d(max_radius,max_radius))+Vector2i(1,1);
    std::vector<Vector2i> indexes;indexes.reserve(64);
    for(int r=low[0];r<high[0];++r)
        for(int c=low[1];c<high[1];++c)
            indexes.push_back(Vector2i(r,c));
    std::sort(indexes.begin(), indexes.end(), [y](Vector2i a, Vector2i b){
        return (a- y).norm2()<(b-y).norm();
    });
    return indexes;
}

double minimum_distance(Vector2d y, Vector2i index, Vector2d delta){
    double a=(delta.pointMultiply(index)             - y).norm2();
    double b=(delta.pointMultiply(index+Vector2i(1,1))- y).norm2();
    double c=(delta.pointMultiply(index+Vector2i(0,1))- y).norm2();
    double d=(delta.pointMultiply(index+Vector2i(1,0))- y).norm2();

    return std::min(std::min(a,b),std::min(c,d) );
}

int Fast2DQuery::find(Vector2d y, double max_radius)
{
    double r=max_radius*max_radius;
    Data best; best.index=-1;
    for(auto index:query_locations(y,max_radius)){
        if(r<minimum_distance(y,index,delta)) continue;
        for(auto& data:grid(index)){
            double dist=(data.x - y).norm2();
            if(dist<r){
                r=dist;
                best=data;
            }
        }
    }
    return best.index;
}

void Fast2DQuery::add(Fast2DQuery::Data data){
    auto y=position(data.x);
    assert(grid.rows>0);
    assert(grid.cols>0);
    y.cap(Vector2i(0,0),Vector2i(grid.rows-1, grid.cols-1));
    grid(y[0],y[1]).push_back(data);
}
}
