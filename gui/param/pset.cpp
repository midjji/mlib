#include <pset.h>
#include <parameter.h>
namespace cvl {
ParamSet::ParamSet(std::string name, std::string desc):name(name),desc(desc){}
std::shared_ptr<ParamSet> ParamSet::create(std::string name, std::string desc){
    return std::make_shared<ParamSet>(name,desc);
}
void ParamSet::set_alive(bool val){
    alive=val;
}
void ParamSet::update_all(){
    for(auto* p:params)
        p->update_value();
}
void ParamSet::add(Parameter* p){
    params.push_back(p);
}
void ParamSet::add(std::shared_ptr<ParamSet> ps){
    subsets_.push_back(ps);
}
const std::vector<std::shared_ptr<ParamSet>>& ParamSet::subsets(){
    return subsets_;
}
std::map<std::string,
std::vector<Parameter*>> ParamSet::param_groups(){
    std::map<std::string,
    std::vector<Parameter*>> groups;
    for(auto* p:params)
        groups[p->group].push_back(p);
return groups;
}


}
