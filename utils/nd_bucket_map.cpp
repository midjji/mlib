#include <mlib/utils/nd_bucket_map.h>
#include <cmath>

namespace cvl{
Fast2DQuery::Fast2DQuery(Vector2d minv,
                         Vector2d maxv, Vector2i buckets):minv(minv),maxv(maxv)
{
    delta_x=std::ceil((maxv[0]-minv[0])/double(buckets[0]));
    delta_y=std::ceil((maxv[1]-minv[1])/double(buckets[1]));
    griddata.resize(buckets[0]*buckets[1],std::vector<Data>()); // essentially holds a pointer in each element
    for(auto& v:griddata) v.reserve(64);
    grid=MatrixAdapter<std::vector<Data>>(&griddata[0],buckets[0],buckets[1]);
}
Vector2i Fast2DQuery::position(Vector2d y){

    Vector2i indexes(int(y[0]/delta_x), int(y[1]/delta_y)); // rounds down...
    return indexes;
}
std::tuple<int,double> find(Vector2d y, std::vector<Fast2DQuery::Data>& datas, double r){
    double minv=r;
    int index=-1;

    for(uint i=0;i<datas.size();++i)
    {

        double val=(datas[i].x-y).norm2();
        if(val<minv){
            minv=val;
            index=i;
        }
    }
    return {index,minv};
}

int Fast2DQuery::find(Vector2d y, double max_radius)
{
    double r=max_radius*max_radius;


    Vector2i low(y[0] - max_radius/delta_x, y[1] - max_radius/delta_y);
    low.cap(Vector2i(0,0),Vector2i(grid.rows, grid.cols));
    Vector2i high(y[0] + std::ceil(max_radius/delta_x)+1, y[1] + std::ceil(max_radius/delta_y)+1);
    high.cap(Vector2i(0,0),Vector2i(grid.rows, grid.cols));

    int index=-1;
    double minv=r;
    for(int r=low[0];r<high[0];++r)
        for(int c=low[1];c<high[1];++c){
            if((low - y).norm2()>r) continue;
            if((high+Vector2i(1,1) - y).norm2()>r) continue;
            auto [i, v]=find(y,grid(y[0],y[1]),r);
                    if(v<minv){
                index=i;
            }
        }
    return index;
}

void Fast2DQuery::add(Fast2DQuery::Data data){
    auto y=position(data.x);
    assert(grid.rows>0);
    assert(grid.cols>0);
    y.cap(Vector2i(0,0),Vector2i(grid.rows-1, grid.cols-1));
    grid(y[0],y[1]).push_back(data);
}
}
