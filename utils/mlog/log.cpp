#include "log.h"
#include <thread>
#include <map>
#include <string>
#include <iostream>

#include <fstream>

#include <regex>
#include <mlib/utils/cvl/syncque.h>
#include <mlib/utils/mlibtime.h>
using std::cout;
using std::endl;
std::string code_position_impl(std::string function, std::string file, int line){
    std::stringstream ss;
    ss<<function<<":"<<line<<"\n";
    return ss.str();
}
namespace cvl{
namespace {
template<class T> std::string str(T t){std::stringstream ss; ss<<t; return ss.str();}
}









struct LogMessage{
    std::string msg, prettyfun, file, line;
    std::thread::id id;
    bool flush;
    float128 time_ns;

    LogMessage()=default;
    LogMessage(std::string msg,
               std::string prettyfun,
               std::string file,
               std::string line,
               std::thread::id id,
               bool flush,
               float128 time_ns):msg(msg),prettyfun(prettyfun),file(file), line(line), id(id),flush(flush),time_ns(time_ns){}

    std::vector<std::string> split_lines() const
    {
        std::vector<std::string> strs;
        strs.push_back("");
        strs.back().reserve(msg.size());



        for(auto m:msg) {
            if(strs.back().size()>0 && strs.back().back()=='\n')
                strs.push_back("");
            strs.back().push_back(m);
        }
        return strs;
    }
};

template<class Key, class Value> bool value_exists(const Value& v, const std::map<Key, Value>& es){
    for(const auto& e:es)
        if(e.second==v)
            return true;
    return false;
}

/**
 * @brief The Log class
 *
 * So this class is part of a a synchronized logging system which lets you log with thread ids,
 *  thread names, pretty function names, etc...
 *
 * However, for this to work, you need to use the mlog macro!
 *
 * mlog()<<...;
 *
 * to configure the logger use:
 *
 * mlog().log->show_function_information(bool);
 * mlog().log->show_threadnames(bool);
 * mlog().log->save_2_file(bool); // default is true, logs to log.txt
 * mlog().log->set_thread_name();
 *
 * you can setup additional logs, using
 * named_mlog("name")
 * which are then independent. and generate files of the type log_<name>.txt
 *
 * logs are independent, and not synchronized,
 * so unless you have some subsystem you dont want generally logged just use the main one.
 *
 *
 *
 *
 * There is a problem with this solution, thread ids are not unique and may be reused.
 * That suxx, but without native threads, there is little to be done.
 * There is a pseudo solution though,
 * you can use mlog()<<clear_name_object
 *
 *
 *
 */
class Log{
    std::ofstream ofs;
    int id=0;
public:
    Log(std::string name):name(name){
        if(name.empty())
            ofs=std::ofstream("log.txt");
        else
            ofs=std::ofstream(name+"_log.txt");
    }
    ~Log(){

        //cout<<"log destructor "<<endl;
        // the log cannot be used in the destructors of global variables
        // its place in the destuction order is unknown and others may wait for it to be destroyed
        // these objects may be filling the log with data, so it must be stopped explicitly.
        std::cout.flush();
        ofs.flush();
    }

    bool log_function_information;
    bool log_thread_information;
    bool save2file=true;
    bool log_time=true;
    void set_thread_name(std::string name_)
    {
        std::unique_lock<std::mutex> ul(log_mtx);
        if(!value_exists(name_,names)){
            names[std::this_thread::get_id()]=name_;
            return;
        }
        // resetting name? warn
        if(names[std::this_thread::get_id()]==name_)
        {
            internal_log(std::string("Warning: void Log::set_thread_name()")+str(0+__LINE__)
                         +" called twice for the same thread id with the same name. "
                         +"Thread id reuse? or copy paste error.\n");
            return;
        }


        // so the name has been used for something else


        // test names in order:
        uint i=0;
        while(value_exists(name_+"_"+str(i++),names));
        names[std::this_thread::get_id()]=name_;        
    }
    void show_function_information(bool val){  std::unique_lock<std::mutex> ul(log_mtx);log_function_information=val;    }
    void show_thread_information(bool val)  {  std::unique_lock<std::mutex> ul(log_mtx);log_thread_information=val;    }



    void log(const LogMessage& msg)
    {
        // perhaps precompute a bunch of stuff, then lock? nah this is cheap!
        std::unique_lock<std::mutex> ul(log_mtx);
        // we always log to cout, but what varies
        internal_log(msg);
    }



    Log(Log const&)             = delete;
    void operator=(Log const&)  = delete;


private:
    void internal_log(const std::string& str){

        std::cout<<str;
        std::cout.flush();
        if(save2file){
            ofs<<id++<<" "<<str;
            ofs.flush();
        }
    }
    void internal_log(const LogMessage& msg){

        std::cout<<get_log_string(msg,false,true,false,false);
        std::cout.flush();
        if(save2file){
            ofs<<id++<<": "<<get_log_string(msg,false,true,false,true);
            ofs.flush();
        }
    }


    // only called from within log which locks!
    std::string get_thread_name(std::thread::id id){
        if(names.find(id)!=names.end())
            return names[id];
        return str(id);
    }

    // only called from within log which locks!
    std::string get_log_string(const LogMessage& msg, bool with_thr_name, bool with_fun_sig, bool with_time, bool with_file){
        // we also collect everything into one pile first,
        //so that we maximize the odds of it getting out in one piece,
        // even if others use std::cout unsynchronized.
        std::vector<std::string> strs=msg.split_lines(); // keeps the '\n'



        std::stringstream ss;
        bool color_log_red=true;

        if(with_file){
            ss<<msg.file<<" ";
        }
        if(color_log_red)
            ss<<"\033[1;31m";



        if(with_time){
            ss<<msg.time_ns<<": ";
            while(ss.str().size()<9+12)
                ss<<" ";
        }
        if(with_thr_name){
            ss<<get_thread_name(msg.id);
            while(ss.str().size()<9+12+10)
                ss<<" ";
        }



        if(with_fun_sig){
            if(with_thr_name)
                ss<<", ";
            ss<<msg.prettyfun<<", "<<msg.line;
        }


        if(color_log_red)
            ss<<"\033[0m";

        if(!ss.str().empty())
            ss<<": ";

        while(ss.str().size()<9+12+10+20)
            ss<<" ";

        // short message:
        if(strs.size()==1 && ss.str().size()+strs[0].size()<120){
            ss<<strs[0];
        }
        else{
            ss<<"\n";
            for(const std::string& str:strs){
                ss<<"    "<<str;
            }
        }
        return ss.str();
    }


    std::mutex log_mtx;


    std::string name;
    std::map<std::thread::id, std::string> names;
    std::map<std::thread::id, uint> ids;



    std::string previous_name="";


};



namespace {
std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds> program_start = std::chrono::steady_clock::now();
std::map<std::string, std::shared_ptr<Log>> logs;
std::mutex logs_mutex;


std::shared_ptr<Log> get_log(std::string log_name){
    std::unique_lock<std::mutex> ul(logs_mutex);

    if(logs.find(log_name)!=logs.end()){
        return logs[log_name];
    }
    return logs[log_name]=std::make_shared<Log>(log_name);
}
}











Logger::~Logger(){    
    assert(log);
    if(ss.str().size()==0) return;
    // this line only adds, it needs to wait for completion!
    log->log(LogMessage(ss.str(),function_description, file, line, std::this_thread::get_id(),flush,time_ns));
}

Logger::Logger(std::string function_description, std::string file, unsigned int line, std::string log_name):
    function_description(function_description), file(file), line(str(line)), log(get_log(log_name)){
    assert(log!=nullptr);
    time_ns= (std::chrono::steady_clock::now() - program_start).count();
}

// dealing with stream control types
// this is the type of std::cout
using CoutType = std::basic_ostream<char, std::char_traits<char> >;
// this is the function signature of std::endl, and most of the other iomanip
// how do I know which one it is? I dont, instead add them to ss as usual. endl will become "\n" though
// meaning we will add a flush to the end if any iomanip has been added.
// A difference from std::cout is that formatting will now be line local. Given that this is for multithreaded enviroments,
// thats a good thing.
typedef CoutType& (*StandardEndLine)(CoutType&);

// define an operator<< to take in std::endl
const Logger& Logger::operator<<(StandardEndLine manip) const
{
    // call the function, but we cannot return it's value
    //manip(std::cout);
    ss<<manip;
    flush=true; // automatic for cout, but not for the file.
    return *this; // lifetime is questionable?
}


void Logger::set_thread_name(const std::string& name){
    log->set_thread_name(name);
}
std::atomic<int> Logger::digits{6};
void Logger::precision(int digits_){
    digits=digits_;
}
void Logger::set_precision_float(){precision(6);}
void Logger::set_precision_double(){precision(12);}
void Logger::set_precision_long_double(){precision(20);}


}
