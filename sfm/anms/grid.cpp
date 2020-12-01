#include <mlib/sfm/anms/grid.h>
#include <iostream>
using std::endl;using std::cout;



namespace cvl{
namespace anms{


Gridlsh::Gridlsh(){}
Gridlsh::Gridlsh(Vector2f minv,
                 Vector2f maxv,
                 uint buckets){
    // I need the function from value to grid pos
    // even buckets sizes makes this simpler
    // ideal bucket count depends on count
    // how to deal with out of scope? asserts
    // MatrixAdapter can only work for D=2
    this->minv=minv;
    this->maxv=maxv;
    this->buckets=buckets;
    delta_x=buckets/(maxv[0]-minv[0]);
    delta_y=buckets/(maxv[1]-minv[1]);
    // lets simplify matters a bit.
    griddata.resize((buckets+1)*(buckets+1),std::vector<Data>()); // essentially holds a pointer in each element
    for(auto& v:griddata) v.reserve(64);
    // rounding errors? lets assume no for now
    grid=MatrixAdapter<std::vector<Data>>(&griddata[0],buckets+1,buckets+1);

}
void Gridlsh::getGridPos(const Vector2f& v, int& row, int& col){
    Vector2f tmp=v-minv;
    // point multiply

    tmp[0]*=delta_x;
    tmp[1]*=delta_y;
    // transformed it to between 0 and 1 times buckets
    tmp[0]=std::floor(tmp[0]);
    tmp[1]=std::floor(tmp[1]);
    row=int(tmp[0]);
    col=int(tmp[0]);

    if(row<0) row=0;
    if(col<0) col=0;
    if(row>=(int)grid.rows) row=grid.rows-1;
    if(col>=(int)grid.cols) col=grid.cols-1;
}
void Gridlsh::getGridPos(const Data& v, int& row, int& col){
    return getGridPos(v.y,row,col);
}
void Gridlsh::add(const Data& v){
    int row,col;
    getGridPos(v,row,col);
    grid(row,col).push_back(v);
}
bool Gridlsh::query(const Data& v,
                    float radius){


    int row,col;
    getGridPos(v,row,col);
    float radius2=radius*radius;

    // check the regular one first, then the others, max prop of finding it there
    if(v.near(grid(row,col),radius2)) return true;


    //compute box for search

    //most distant neg point
    int rowmin,colmin,rowmax,colmax;
    getGridPos(v.y-Vector2f(radius,radius),rowmin,colmin);
    getGridPos(v.y+Vector2f(radius,radius),rowmax,colmax);
    //rowmin=std::max(0,rowmin-2);
    //colmin=std::max(0,colmin-2);
    //rowmax=std::min((int)grid.rows-1,rowmax+2);
    //colmax=std::min((int)grid.cols-1,colmax+2);



    for(int r=rowmin;r<=rowmax;++r)
        for(int c=colmin;c<=colmax;++c){

            assert(r<(int)grid.rows);
            assert(c<(int)grid.cols);
            assert(r>-1);
            assert(c>-1);

            if(r==row && c==col) continue;
            if(v.near(grid(r,c),radius2)) return true;
        }
    return false;
}


void GridSolver::init(const std::vector<Data>& datas,const std::vector<Data>& locked){


    //Solver::init(datas,locked);// can I call the overloaded base function, probably yes...
    {
    inited=true;
    this->datas=datas;
    filtered.reserve(datas.size());
    filtered.clear();
    // the locked are added to filtered directly by default, but latter need to be tested against them too so init and compute may need to be overloaded
    filtered=locked;
    filtered.reserve(datas.size()+locked.size());

    if(!issorted(datas))
        sort(this->datas.begin(), this->datas.end(),[](const Data& lhs, const Data& rhs) { return lhs.str>rhs.str; });
    }
    if(datas.size()==0) return;
    // find the spans

    float minx=datas.at(0).y[0];
    float miny=datas.at(0).y[1];
    float maxx=datas.at(0).y[0];
    float maxy=datas.at(0).y[1];
    for(auto data:datas){
        if(data.y[0]<minx) minx=data.y[0];
        if(data.y[0]>maxx) maxx=data.y[0];
        if(data.y[1]<minx) miny=data.y[1];
        if(data.y[1]>maxx) maxy=data.y[1];
    }
    for(auto data:locked){
        if(data.y[0]<minx) minx=data.y[0];
        if(data.y[0]>maxx) maxx=data.y[0];
        if(data.y[1]<minx) miny=data.y[1];
        if(data.y[1]>maxx) maxy=data.y[1];
    }

    grid=Gridlsh(Vector2f(minx,miny),Vector2f(maxx,maxy),20);

    // locked should be added to grid without checking distances!
    // locked has already been added to filtered
    assert(filtered.size()==locked.size());
    for(const Data& data:locked){
        grid.add(data);
    }
}
void GridSolver::init(const std::vector<Data>& datas,const std::vector<Data>& locked, float minx, float miny, float maxx, float maxy ){

    //Solver::init(datas,locked);// can I call the overloaded base function, probably yes...
    {
    inited=true;
    this->datas=datas;
    filtered.reserve(datas.size());
    filtered.clear();
    // the locked are added to filtered directly by default, but latter need to be tested against them too so init and compute may need to be overloaded
    filtered=locked;
    filtered.reserve(datas.size()+locked.size());

    if(!issorted(datas))
        sort(this->datas.begin(), this->datas.end(),[](const Data& lhs, const Data& rhs) { return lhs.str>rhs.str; });
    }

    grid=Gridlsh(Vector2f(minx,miny),Vector2f(maxx,maxy),40);

    // locked should be added to grid without checking distances!
    // locked has already been added to filtered
    assert(filtered.size()==locked.size());
    for(const Data& data:locked){
        grid.add(data);
    }
}

void GridSolver::compute(double minRadius, int minKeep){
    assert(inited);
    if(datas.size()==0) return;

    // two options either add once and query alot,
    // or add many and query once.

    // add once

    for(uint i=0;i<datas.size();++i){
        Data d=datas[i];

        int remaining=(datas.size()-i);
        if(minKeep>0 && (int)filtered.size()+remaining<=minKeep){

            filtered.push_back(d);
        }else{

            if(!grid.query(d,minRadius)){

                grid.add(d);
                filtered.push_back(d);
            }
        }
    }
}



} // end namespace anms
}// end namespace cvl
