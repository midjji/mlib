#include <parameter.h>
namespace cvl {


Parameter::Parameter(std::string name,
          std::string group,
          std::string desc):
    name(name),group(group),desc(desc) {}
bool Parameter::is_int() const{return false;};
bool Parameter::is_double() const{return false;};
}

