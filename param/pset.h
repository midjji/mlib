#pragma once
#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <memory>

namespace cvl {
class Parameter;
class ParamSet{
public:
    std::string name;
    std::string desc;
    ParamSet(std::string name="unnamed", std::string desc="no desc for pset");
    static std::shared_ptr<ParamSet> create(std::string name="unnamed", std::string desc="no desc for pset");
    // "" is the default, for the top level...
    std::map<std::string,
    std::vector<Parameter*>> param_groups();
    // sub parameter sets,

    void set_alive(bool);
    void update_all();
    void add(Parameter* p); // replace by templated factory?
    void add(std::shared_ptr<ParamSet> ps);
    const std::vector<std::shared_ptr<ParamSet>>& subsets();
private:
    std::vector<Parameter*> params;
    std::atomic<bool> alive;
    std::vector<std::shared_ptr<ParamSet>> subsets_;
};
using ParamSetPtr=std::shared_ptr<ParamSet>;

}
