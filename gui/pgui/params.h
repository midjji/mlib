#pragma once
#include <string>
#include <vector>
struct IntParameter{
    int val;

};
class Parameter{
    virtual std::string name();
};

class ParamSet{
public:
    std::string name;
    std::string desc;
    ParamSet(std::string name="unnamed"):name(name){}   
    std::vector<Parameter*> ps;
    std::vector<ParamSet*> pss;

};
