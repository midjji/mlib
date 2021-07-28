#include "mlib/utils/configuration.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <exception>

#include "mlib/utils/string_helpers.h"

#include <filesystem>
using std::cout;using std::endl;
namespace mlib{

Configuration::Configuration(std::string configurationfile){
    this->path=configurationfile;
    if(std::filesystem::exists(configurationfile))
        cout<<"Reading conf: "<<this->path<<endl;
    else
        cout<<"Creating conf: "<<this->path<<endl;


}
std::string Configuration::getOutputDirectory(){
    return getStr("OutputDirectory","./","");
}
std::string Configuration::getDataDirectory(){
    return getStr("DataDirectory","./","");
}
std::string Configuration::getStr(std::string name, std::string value, std::string comment){
    init();
    if(params.find(name) != params.end())
        return (*params.find(name)).second;
    cout<<"not found"<<endl;
    // save the parameter to the config file
    std::ofstream fos;
    fos.open(path.c_str(), std::ios::app);
    fos<<name<<" "<<value<<" #"<<comment<<std::endl;
    fos.close();
    params[name]=value;
    return value;
}


void Configuration::init()
{
    if(inited)
        return;
    inited=true;
    if(!std::filesystem::exists(path)) cout<<path<<endl;


    std::ifstream fin(path.c_str());

    // not open directly after read means it does not exist... create it
    if(!fin){
        fin.close();
        cout<<"path"<<path<<endl;
        std::ofstream fout(path.c_str());
        fout<<"# Configurationsfile\n" <<
              "# Assumed UTF8\n"<<
              "# CaSe SensitivE!\n"<<
              "# rows beginning with # will be ignored\n"<<
              "# All parameters must have a unique name\n"<<
              "# no parameter name or value may contain whitespace\n"<<
              "# \"name value comment\" the \"name value\" ' ' is important ans is the \"value comment\" ' '!\n"<<
              "# strings without whitespacem, ints doubles && bools are supported\n"<<
              "# bools are written as true= 1,false =0\n "<<
              "# examples of correctly formatted lines follow.\n"<<                  "\n"<<
              "#when the system asks for a parameter value it is expected to provide a default should it be missing from the configuration file.\n"<<                  "\n"<<
              "#name value comment\n"<<
              "namn varde #kommentar the '#' here isnt special but it would be good practice\n"<<
              "# Parametrarna\n"<<
              "DataDirectory ./ # the base dataset directory\n"<<
              "OutputDirectory ./ # the base output directory\n"<<
              "# Forgotten\n"<<std::endl;
        fout.close();
        fin.open(path.c_str());
        cout<<"path= \'"<<path<<std::endl;
        assert(fin && "created file has to exist! check permissions");
    }



    std::string row;
    std::string key;
    std::string value;
    while(fin.good())   // if file dosent exist fin.good=false...
    {
        getline(fin ,row,'\n');
        if(row.size()!=0)
        {
            if(row.substr(1,0)!="#")
            {
                //assumes ' ' separated
                std::stringstream ss(row);
                ss>> key;
                ss>> value;
                params[key]=value;
            }
        }
    }
    fin.close();
    inited=true;
}


} // end namespace mlib
