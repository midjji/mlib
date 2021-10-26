#pragma once
#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <memory>

namespace cvl {
class Parameter;
class PSet{
public:
    std::string name;
    std::string desc;
    PSet(std::string name="unnamed", std::string desc="no desc for pset");
    static std::shared_ptr<PSet> create(std::string name="unnamed", std::string desc="no desc for pset");
    // "" is the default, for the top level...

    std::map<std::string,
    std::vector<std::shared_ptr<Parameter>>> param_groups();
    // sub parameter sets,

    void set_alive(bool);
    void update_all();

    template<class T, class... Args>
    T* add(std::string name, // must be unique for pset!
                   Args... args){
        auto it=name2param.find(name);
        if(it!=name2param.end()) return (T*)it->second.get();
        auto p=std::make_shared<T>(args...);
        name2param[name]=p;
        return (T*)it->second.get();
    }
    void add(std::shared_ptr<PSet> ps);
    const std::vector<std::shared_ptr<PSet>>& subsets();

    std::string serialize();
    static PSet deserialize();
private:
    //the parameters by name, which is unique!
    std::map<std::string, std::shared_ptr<Parameter>> name2param;

    std::atomic<bool> alive;
    std::vector<std::shared_ptr<PSet>> subsets_;
};
using PSetPtr=std::shared_ptr<PSet>;

}

