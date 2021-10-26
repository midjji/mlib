#include <pset.h>
#include <parameter.h>
#include <sstream>
namespace cvl {
PSet::PSet(std::string name, std::string desc):name(name),desc(desc){}
std::shared_ptr<PSet> PSet::create(std::string name, std::string desc){
    return std::make_shared<PSet>(name,desc);
}
void PSet::set_alive(bool val){
    alive=val;
}
void PSet::update_all(){
    for(auto& [name, p]:name2param)
        p->update_value();
}
void PSet::add(std::shared_ptr<PSet> ps){
    subsets_.push_back(ps);
}
const std::vector<std::shared_ptr<PSet>>& PSet::subsets(){
    return subsets_;
}
std::map<std::string,
std::vector<std::shared_ptr<Parameter>>> PSet::param_groups(){
    std::map<std::string,
    std::vector<std::shared_ptr<Parameter>>> groups;
    for(auto [name, p]:name2param)
        groups[p->group].push_back(p);
return groups;
}


std::string PSet::serialize(){
    // we want to be able to save and load with missing values.
    // we do this by requireing each param to have a unique name,
    // but only in the chain?

/*
    std::stringstream ss;
    ss<<"#"<<name<<"\n";
    auto pgroups=param_groups();
    for(auto [group, ps]:pgroups) {
            std::stringstream ssp;
            ssp<<"#"<<group<<"\n";


    }
*/


}
static PSet deserialize();

}
