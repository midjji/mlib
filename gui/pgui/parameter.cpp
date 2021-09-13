#include <parameter.h>
#include <QWidget>
namespace cvl {


Parameter::Parameter(std::string name,
          std::string group,
          std::string desc):
    name(name),group(group),desc(desc) {}

QWidget* Parameter::display() const{return nullptr;}
}
