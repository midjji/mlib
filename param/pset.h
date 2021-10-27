#pragma once
#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <memory>
#include <mlib/utils/mlog/log.h>

namespace cvl {
class Parameter;



class PSet{

public:
    std::string name;
    std::string desc;
    PSet(std::string display_name="unnamed",
         std::string desc="a pset");

    // "" is the default, for the top level...

    std::map<std::string,
    std::vector<std::shared_ptr<Parameter>>> param_groups();
    // sub parameter sets,
    void update_all();

    template<class T, class... Args>
    std::shared_ptr<T> param(std::string unique_identifier, // unique for this
               Args... args)
    {
        auto it=identifier2param.find(unique_identifier);
        if(it!=identifier2param.end())
        {
             std::shared_ptr<T> fit=std::dynamic_pointer_cast<T>(it->second);
             if(!fit){
                 mlog()<<"bad user\n";
             }
            return fit;
        }
        auto p=std::make_shared<T>(args...);
        identifier2param[unique_identifier]=p;
        return p;
    }
    // its a question of when what information is known.
    void add(std::string unique_identifier,
             std::shared_ptr<PSet> ps);
    std::vector<std::shared_ptr<PSet>> subsets();

//TODO fix save, load...

    std::string display()const;
private:
    //the parametersb
    std::map<std::string, std::shared_ptr<Parameter>> identifier2param;
    std::map<std::string, std::shared_ptr<PSet>> identifier2subset;
};
using PSetPtr=std::shared_ptr<PSet>;

}

