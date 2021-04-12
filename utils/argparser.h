#pragma once
/* ********************************* FILE ************************************/
/** \file    argparser.h
 *
 * \brief
 *
 * \remark
 * - c++11
 * - no dependencies
 *
 * Initialization using a rotation matrix is allowed but not ideal. Such conversions can give errors if the input isnt a rotation.
 *
 * \todo
 *  - better type specification,
 *  - auto validation of input set
 *  - ranges
 *  - variable input count
 *
 *
 *
 *
 * \author   Mikael Persson
 * \date     2014-04-01
 * \note MIT licence
 *
 ******************************************************************************/

#include <string>
#include <vector>
#include <map>
namespace mlib{
class Command{
public:
    enum type{String, Double, Bool};
    Command()=default;
    Command(std::string name,
            int count,
            std::string Default="",
            std::string desc="",
            bool required=false);
    std::string name="";
    std::string desc="";
    int count=0; // number of arguments
    int num_times_in_cmd_line=0;
    bool required=false;
    type t;

// what the user provided or default
    std::vector<std::string> inputs;

    bool to_bool();
    double to_double();


};

std::ostream& operator<<(std::ostream& os, Command cmd);
class ArgParser{
public:
    bool parse_args(int argc, char** argv);
    bool parse_args(std::vector<std::string> args);

    void add_option(std::string name,
                    int count,
                    std::string Default="",
                    std::string desc="",
                    bool required=false);
    // unnamed arguments must be in order and first! and also size one
    void add_parameter(std::string name, std::string desc, std::string Default);

    void add_option(Command cmd);



    // program parameter0 parameter1 ... --option0 sdf sdf --option2 one
    std::vector<Command> parameters;
    // all options can always be queried,
    // unless provided, they give the default!
    std::map<std::string, Command> options;
    std::string program_name;

    bool is_set(std::string name);
    std::vector<std::string> get_args(std::string name);
    std::string get_arg(std::string name);
    double get_double_arg(std::string name);
    bool get_bool_arg(std::string name);

    // next parameter, either as double or str,
    uint parameter_index=1; // program name not included!
    double param_double();
    bool param_bool();

    bool args_parsed=false;
    void help();
};
}
