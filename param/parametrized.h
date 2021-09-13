#pragma once
#include <pset.h>
#include <int_parameter.h>
#include <real_parameter.h>

namespace cvl {
struct Parametrized {
    Parametrized(std::string name="unnamed",
                 std::string desc="no desc");
    virtual ~Parametrized();

    std::shared_ptr<ParamSet> params();
    void update_all();

    IntParameter* pint(
            int default_value,
            std::string name="unnamed",
            std::string group="",
            std::string desc="no tool tip",
            int minv=std::numeric_limits<int>::min(),
            int maxv=std::numeric_limits<int>::max());

    RealParameter* preal(double default_value,
                         std::string name="unnamed",
                         std::string group="",
                         std::string desc="no tool tip",
                         double minv=std::numeric_limits<double>::lowest(),
                         double maxv=std::numeric_limits<double>::max());
private:
    std::shared_ptr<ParamSet> param;
};

}

