#include <pset.h>
#include <parameter.h>
namespace cvl {
PSet::PSet(std::string name, std::string desc):name(name),desc(desc){}
std::shared_ptr<PSet> PSet::create(std::string name, std::string desc){
    return std::make_shared<PSet>(name,desc);
}
void PSet::set_alive(bool val){
    alive=val;
}
void PSet::update_all(){
    for(auto* p:params)
        p->update_value();
}
void PSet::add(Parameter* p){
    params.push_back(p);
}
void PSet::add(std::shared_ptr<PSet> ps){
    subsets_.push_back(ps);
}
const std::vector<std::shared_ptr<PSet>>& PSet::subsets(){
    return subsets_;
}
std::map<std::string,
std::vector<Parameter*>> PSet::param_groups(){
    std::map<std::string,
    std::vector<Parameter*>> groups;
    for(auto* p:params)
        groups[p->group].push_back(p);
return groups;
}


}
