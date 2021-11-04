#include <pset.h>
#include <parameter.h>
#include <sstream>
#include <filesystem>

namespace cvl {
PSet::PSet(std::string name,
           std::string desc):name(name),desc(desc){}


void PSet::update_all()
{
    for(auto& [name, p]:identifier2param)
        p->update_value();
}
void PSet::add(std::string unique_identifier,
               std::shared_ptr<PSet> ps){
    identifier2subset[unique_identifier]=ps;
}

std::map<std::string,
std::vector<std::shared_ptr<Parameter>>> PSet::param_groups(){
    std::map<std::string,
    std::vector<std::shared_ptr<Parameter>>> groups;
    for(auto& [name, p]:identifier2param)
        groups[p->group].push_back(p);
return groups;
}
std::vector<std::shared_ptr<PSet>> PSet::subsets(){
    std::vector<std::shared_ptr<PSet>> ss;ss.reserve(identifier2subset.size());
    for(auto& [uid, ps]:identifier2subset)ss.push_back(ps);
    return ss;
}
std::string PSet::display()const {
    std::stringstream ss;
    ss<<"PSet: "<<name;
    for(auto& [uid, p]:identifier2param)
        ss<<uid<<": "<<p->display()<<"\n";
    for(auto& [uid, ps]:identifier2subset)
        ss<<"Subset: "<<uid<<" "<<ps->display()<<"\n";
    return ss.str();
}




void PSet::load(std::string path)
{
#if 0
    std::ifstream ifs(path+"/"+name+".txt");
    int params=0;
    ifs>>params;
    for(int i=0;i<params;++i)
    {
        std::string uid;
        char toss;
        int count=0;
        ifs>>count;
        ifs>>toss;
        for(int i=0;i<count;++i){
            ifs>>toss;
            uid.push_back(toss);
        }
        ifs>>Parameter
    }
#endif

}
void PSet::save(std::string path)
{
#if 0
    std::filesystem::create_directories(path);
    std::ofstream ofs(path+name+".txt");
    ofs<<identifier2param.size()<<" ";
    for(auto& [uid, p]: identifier2param)
    {
        ofs<<uid.size()<<" "<<uid<<p->serialize();
    }
    for(auto& [uid, ps]:identifier2subset){

    }
#endif
}


}




