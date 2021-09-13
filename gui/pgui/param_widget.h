#pragma once
#include <string>
#include <vector>
class QWidget;

namespace cvl {
class Parameter;
QWidget* display(Parameter* p, QWidget* parent);
QWidget* display_group(std::vector<Parameter*> group, std::string groupname, QWidget* parent);
}
