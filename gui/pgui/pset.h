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
    ParamSet(std::string name="unnamed");
    // "" is the default, for the top level...
    std::map<std::string,
    std::vector<Parameter*>> param_groups();
    // sub parameter sets,
    std::vector<std::shared_ptr<ParamSet>> sub_psets;
private:
    std::vector<Parameter*> params;
    std::atomic<bool> alive;

};
}
