#pragma once
#include <string>
#include <vector>
#include <memory>
class QWidget;

namespace cvl {
class Parameter;
QWidget* display(Parameter* p, QWidget* parent);
QWidget* display_group(std::vector<std::shared_ptr<Parameter>> group, std::string groupname, QWidget* parent);
}
