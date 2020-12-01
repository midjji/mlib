/**
 * @file
 * @author Mikael Persson <mikael.p.persson@gmail.com>
 * @version 1.0
 * \date 2013-05-10:13-45-43
 * @section LICENSE Mit licence
 *
 * Copyright Mikael Persson.
 *
 * @section DESCRIPTION
 * Header for the configuration class
 *
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <exception>



namespace mlib{



/**
 * @brief The Configuration class - Simplified runtime configuration.
 *
 *
 * # Configurationsfile
 * # Assumed UTF8
 * # CaSe SensitivE!
 * # rows beginning with # will be ignored
 * # All parameters must have a unique name
 * # no parameter name or value may contain whitespace
 * # \"name value comment\" the \"name value\" ' ' is important as is the \"value comment\" ' '!
 * # strings without whitespacem, ints doubles && bools are supported
 * # bools are written as true= 1,false =0
 * # examples of correctly formatted lines follow. "
 * #when the system asks for a parameter value it is expected to provide a default should it be missing from the configuration file. "
 * #name value comment
 * namn varde #kommentar the '#' here isnt special but it would be good practice
 * # Parametrarna
 * DataDirectory ./ # the base dataset directory
 * OutputDirectory ./ # the base output directory
 * # Forgotten
 *
 *
 *
 *
 *
 *
 *
 */
class Configuration {
public:

    Configuration(std::string configurationfile="configuration.ini");

    std::string get(std::string name, const char* val, std::string comment="");


    /**
 * @brief get
 * @param name - the name of the config option
 * @param value - the value of the config option
 * @param comment - comment which should be added at the end if missing
 * @return
 */
    template<class T> T get(std::string name, T value, std::string comment=""){
        assert(name.size()>0);
        T ret;
        std::stringstream ss;
        ss<<value;
        std::string s=ss.str();
        s=getStr(name,s,comment);
        std::stringstream sv(s);
        sv >> ret;
        return ret;
    }



    void set(std::string name, std::string value){
        init();
        assert(name.size()>0);
        params[name]=value;
    }

    // convenience
    std::string getDataDirectory();
    std::string getOutputDirectory();

private:
    bool inited=false;
    std::string path;
    /**
 * Parses the configuration file or creates it if it is missing.
 * does nothing if its already been called
 */
    void init();
    std::map<std::string,std::string> params;
    std::string getStr(std::string name, std::string val, std::string comment="");


};





} // end namespace mlib
