#include <mlib/sfm/anms/base.h>
#include <iostream>

namespace cvl{
namespace anms{



Data::Data(){}
Data::Data(float str, float x, float y, int id):str(str),y(cvl::Vector2f(x,y)),id(id){}
bool Data::near(const std::vector<Data>& datas,float radius2) const{

    for(const Data& data:datas)
        if((data.y-y).squaredLength()<=radius2)
            return true;
    return false;
}

void Solver::init(const std::vector<Data>& datas){
    init(datas,std::vector<Data>());
}
void Solver::init(const std::vector<Data>& datas,
                  const std::vector<Data>& locked){
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

void Solver::compute(double minRadius,int minKeep){
    assert(inited);
    assert(minRadius>0);
    if(datas.size()<2) {
        filtered=datas;
        return;
    }

    float radius2=minRadius*minRadius;
    for(uint i=0;i<datas.size();++i){
        Data d=datas[i];

        int remaining=int((datas.size()-i));
        if(minKeep>0 && (int)filtered.size()+remaining<=minKeep){
            filtered.push_back(d);
        }else{
            if(!d.near(filtered,radius2))
                filtered.push_back(d);
        }
    }
}
bool Solver::exact(){return true;}


bool issorted(const std::vector<Data>& datas){
    for(uint i=1;i<datas.size();++i)
        if(datas[i-1].str<datas[i].str) return false;
    return true;
}




std::vector<int> getIds(std::vector<anms::Data>& datas){
    std::vector<int> ids;ids.reserve(datas.size());
    for(auto d:datas)
        ids.push_back(d.id);

    return ids;
}

std::vector<float> getStrengths(std::vector<anms::Data>& datas){
    std::vector<float> strs;strs.reserve(datas.size());
    for(auto d:datas)
        strs.push_back(d.str);

    return strs;
}










} // end namespace anms
}// end namespace cvl
