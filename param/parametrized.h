#pragma once
#include <mlib/param/pset.h>
#include <mlib/param/int_parameter.h>
#include <mlib/param/real_parameter.h>

namespace cvl {

struct Parametrized {
    // parametrized creates a public dependency, not ideal...
    // parametrized is a bad idea...
   // we want to abstract this further...


    Parametrized(std::string name="unnamed",
                 std::string desc="no desc");
    virtual ~Parametrized();


    void add(std::string unique_identifier,
             std::shared_ptr<PSet> p);
    std::shared_ptr<PSet> params();
    void update_all();

    // could these be int pointers, and so on? yeah, it would simply be the latest value
    // consider refactor to that for a cool boost to modularity!
    // this class owns,
    // these are implicitly shared, but for performance reasons on the query side,
    // a normal pointer is returned.
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
    std::string display()const {
        return param->display();
    };

private:
    std::shared_ptr<PSet> param;
};

}

