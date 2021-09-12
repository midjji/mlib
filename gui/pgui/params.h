#pragma once
#include <string>
#include <vector>
#include <map>
class Parameter{
public:
    enum type_t { integer_range,
                 real_range,
               options, path };
    Parameter(type_t type,
              std::string name="unnamed",
              std::string desc="no tool tip"):
        type(type),name(name) {}
    std::string name;
    std::string desc; // tool tip
    type_t type;
};

struct IntRangeParameter:public Parameter{
    int val;
    IntRangeParameter():Parameter(integer_range){};
};



class ParamSet{
public:
    std::string name;
    std::string desc;
    ParamSet(std::string name="unnamed"):name(name){}   
    // "" is the default, for the top level...
    std::map<std::string, std::vector<Parameter*>> param_groups;
    std::vector<ParamSet*> pss;

};
