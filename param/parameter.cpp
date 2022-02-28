#include <parameter.h>
#include <int_parameter.h>
#include <real_parameter.h>
namespace cvl {


Parameter::Parameter(std::string name,
          std::string group,
          std::string desc):
    name(name),group(group),desc(desc) {}
Parameter::~Parameter(){}
bool Parameter::is_int() const{return false;};
bool Parameter::is_double() const{return false;};
std::string Parameter::display() const{return "";};

}

