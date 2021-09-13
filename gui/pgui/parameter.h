#pragma once
#include <string>

class QWidget;
namespace cvl {


/**
 * @brief The Parameter class
 * Privately depends on qt5, may not publically depend on it
 *
 */
class Parameter{
public:
    Parameter(std::string name,
              std::string group,
              std::string desc);

    virtual QWidget* display() const;
    const std::string name;
    const std::string desc; // tool tip
    const std::string group;
    virtual bool update_value()=0;
};
}
