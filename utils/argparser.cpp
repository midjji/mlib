#include <mlib/utils/argparser.h>
#include <mlib/utils/mlog/log.h>
#include <iostream>
#include <cassert>
#include <queue>
#include <sstream>
#include <set>
using std::cout;
using std::endl;

std::vector<std::string> args(int argc, char** argv) {
    std::vector<std::string> ar;ar.reserve(argc);
    for(int i=0;i<argc;++i){
        ar.push_back(argv[i]);
    }

    return ar;
}
std::map<std::string, std::string> args(int argc, char** argv,std::vector<std::tuple<std::string, std::string>> name2default)
{

    std::set<std::string> names;
    for(const auto& [n,toss]:name2default){
        auto it=names.find(n);
        if(it!=names.end()) {
            mlog()<<"repeated parameter name: \""<<n<<"\" last will be used. \n";
        }
    }

    std::map<std::string, std::string> rets;
    auto ar=args(argc,argv);
    if(ar.size()==0)
    {
        mlog()<<"empty arguments, defaults will be used. \n";
        rets["program_name"]="unknown";
    }
    else
    {
        rets["program_name"]=ar[0];
    }

    for(int i=0;i<int(name2default.size());++i)
    {
        auto [name, val]=name2default[i];
        if(i+1<int(ar.size()))
            rets[name]=ar[i+1];
        else
            rets[name]=val;
    }
    return rets;
}
namespace mlib{
std::vector<std::string> split(const std::string& args)
{

    std::vector<std::string> words;words.reserve(args.size());
    if(args.empty())
        return words;

    std::string tmp;
    for(const auto& arg:args){
        if(arg==' '){
            if(tmp.size()>0)
                words.push_back(tmp);
            tmp.clear();
        }
        else
            tmp.push_back(arg);
    }
    if(!tmp.empty())
        words.push_back(tmp);
    return words;
}

Command::Command(std::string name,
                 int count,
                 std::string Default,
                 std::string desc,
                 bool required):
    name(name),desc(desc), count(count),required(required),inputs(split(Default)){
    //cout<<"default: "<<Default<<split(Default).size()<<endl;

}
bool Command::to_bool(){
    std::string str=inputs[0];
    // techincally covered in lower case by ss if configured right, but lets be clear
    if(str==std::string("true")) return true;
    if(str==std::string("false")) return false;
    if(str==std::string("True")) return true;
    if(str==std::string("False")) return false;
    if(str==std::string("y")) return true;
    if(str==std::string("n")) return false;
    std::stringstream ss(str);
    double d;
    ss>>d;
    if(!ss) cout<<"failed to parse: \""<<str<<"\" "<<*this<<"\n";
    return d;
}
double Command::to_double(){
    std::string str=inputs[0];
    std::stringstream ss(str);
    double d;
    ss>>d;
    if(!ss) cout<<"failed to parse: \""<<str<<"\" "<<*this<<"\n";
    return d;
}
std::ostream& operator<<(std::ostream& os, Command cmd){

    return os<<cmd.name<<" "<<cmd.desc;
}
void ArgParser::add_parameter(std::string name,
                              std::string desc,
                              std::string Default){
    if(args_parsed){
        std::cerr<<"trying to add an argument after parsing is complete"<<endl;
        exit(1);
    }
    if(parameters.size()==0)
        parameters.push_back(Command("program name",1,"name of the app","program",true));
    parameters.push_back(Command(name,1,Default,desc,true));
}
void ArgParser::add_option(Command cmd){
    if(args_parsed){
        std::cerr<<"trying to add an argument after parsing is complete"<<endl;
        exit(1);
    }
    assert(options.find(cmd.name)==options.end());
    options[cmd.name]=cmd;
}
void ArgParser::add_option(std::string name,
                           int count,
                           std::string Default,
                           std::string desc,
                           bool required){
    return add_option(Command(name,count,Default,desc,required));
}


bool ArgParser::is_set(std::string name){

    auto it=options.find(name);
    if(it==options.end()){
        mlog()<<"Warning: Asked for non existent command line option: \""<<name<<"\""<<endl;
        return false;
    }

    return (it->second.num_times_in_cmd_line>0);
}

std::vector<std::string> ArgParser::get_args(std::string name){

    return options.find(name)->second.inputs;
}
std::string ArgParser::get_arg(std::string name){

    return get_args(name).at(0);
}
double ArgParser::get_double_arg(std::string name){
    auto it=options.find(name);
    if(it==options.end())
        mlog()<<"option not found: "+name<<endl;
    return it->second.to_double();
}
bool ArgParser::get_bool_arg(std::string name){
    auto it=options.find(name);
    if(it==options.end())
        mlog()<<"option not found: "+name<<endl;
    return it->second.to_bool();
}
double ArgParser::param_double(){
    if(!args_parsed) {std::cerr<<"asking for args without having parsed any!"<<endl;exit(1);}
    if(!(parameter_index<parameters.size())){cout<<"too few parameters, asking for double: "<<parameter_index<<" of "<<parameters.size()<<endl;}
    Command cmd=parameters.at(parameter_index++);
    return cmd.to_double();

}
bool ArgParser::param_bool(){
    if(!args_parsed) {std::cerr<<"asking for args without having parsed any!"<<endl;exit(1);}
    if(!(parameter_index<parameters.size())){cout<<"too few parameters, asking for bool: "<<parameter_index<<" of "<<parameters.size()<<endl;}
    Command cmd=parameters.at(parameter_index++);
    return cmd.to_bool();
}

bool ArgParser::parse_args(std::vector<std::string> args){
    args_parsed=true;

    uint i=0;
    for(auto& p:parameters){
        if(i<args.size())
            p.inputs={args[i++]};
    }


    std::queue<std::string> strings;
    for(uint i=parameters.size();i<args.size();++i)
        strings.push(args[i]);



    // now look for predefined commands, with a predefined number of options
    while(strings.size()>0)
    {
        std::string arg=strings.front();
        strings.pop();
        auto search=options.find(arg);
        if(search==options.end()){
            cout<<"argument: "<<arg<<" not found"<<endl; continue;
        }
        if(search->second.num_times_in_cmd_line++>1){
            mlog()<<"Warning: duplicate option in command line \""<<arg<<"\". Overiding earlier values"<<endl;
            return false;
        }





        std::vector<std::string> cmd_args;
        // found the command, get number of args.
        if(search->second.count<0){ // means infinite
            while(strings.size()>0){
                cmd_args.push_back(strings.front());
                strings.pop();
            }
            break;
        }
        if(strings.size()<(uint)search->second.count){
            cout<<"cmd: "<<arg<< " has too few arguments, requires: "<<search->second<<" found: "<<strings.size()<<endl;
            break;
        }
        for(int i=0;i<search->second.count;++i){
            cmd_args.push_back(strings.front());
            strings.pop();
        }
        search->second.inputs=cmd_args;
    }
    // check that all required args are found!

    // all options have a default.

    bool good=true;
    for(const auto& opt:options){
        auto it=options.find(opt.second.name);
        if(it!=options.end()) continue;

        if(opt.second.required)
        {
            good =false;
            cout<<"Critical argument missing: "<<opt.second.name<<endl;
        }
    }
    if(!good)
        help();
    args_parsed=true;
    return good;
}
bool ArgParser::parse_args(int argc, char** argv){
    std::vector<std::string> args;
    for(int i=0;i<argc;++i)
        args.push_back(argv[i]);
    return parse_args(args);
}
void ArgParser::help(){
    //
    cout<<"Available commands"<<endl;
    for(const auto& cmd:options)
        //cout<<"    "<<cmd.first<< " "<<cmd.second<<endl;
        cout<<" "<<cmd.second<<endl;
    cout<<"got commands: "<<endl;
    for(const auto& cmd:options){
        //cout<<"    "<<cmd.first;
        for(const auto& arg:cmd.second.inputs)
            cout<<" "<<arg;
        cout<<endl;
    }
}
}
