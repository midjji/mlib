#pragma once
#include <string>
#include <vector>



class Param
{
    std::string name;
    std::string desc;
    // something that tells it how to add it to the widget?
    // doit(widget); // ideally without requireing widget to be known... hmm...
};
class ParamSet
{
    std::string name;
    std::string desc;
    std::vector<Param*> params;
    std::vector<ParamSet*> children;

    void add_param(Param* param){
        params.push_back(param);
    }
    void add_child(ParamSet* child){
        children.push_back(child);
    }
    virtual void parse_changes() =0;
bool changed=false;
};

/*
template<class T>
class Range{
    T min, max, val;
    Range(T min, T max, T val):min(min),max(max),val(val){}
    // explicit cast to type?
    T get(){return val;}
    T set(T t){
        if(t<min) t=min;
        if(t>max) t=max;
        val=t;
        return val;
    }
};


template<class Type> // most commonly Range
class TypedParam:public Param{
    Type t;
    TypedParam(){}
};



template<class T> class TypedParamSet:public ParamSet{
   void parse_changes(T& t){
        if(!changed)
            return;
        for(auto param:params){
            t.(*param.callback); //std::invoke?
        // would be std::invoke(t,param.callback)
    }
};
*/
