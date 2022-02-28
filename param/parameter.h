#pragma once
#include <string>
#include <sstream>
#include <memory>
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
    virtual ~Parameter();
    const std::string name;
    const std::string desc; // tool tip
    const std::string group;
    virtual bool update_value()=0;
    enum type_t{integer, real, options};
    virtual type_t type() const=0;
    virtual bool is_int() const;
    virtual bool is_double() const;
    virtual std::string display() const;
    //std::string serialize() const=0;
    //std::shared_ptr<Parameter> read(std::istream& is);

};

}
