#pragma once
#include <parameter.h>
#include <atomic>
#include <mutex>

namespace cvl {
// Not templated in order to facilitate hiding implementation!
struct RealParameter:public Parameter {
    RealParameter(
            double default_value,
            std::string name="unnamed",
            std::string group="",
            std::string desc="no tool tip",
            double minv=std::numeric_limits<double>::lowest(),
            double maxv=std::numeric_limits<double>::max());
    // the value range
    const double minv;
    const double maxv;
    bool ranged() const;
    // USED BY THE Parametrized class...
    // the user value
    double value() const;
    // the user selects when to update
    bool update_value() override;
    type_t type() const override;

    //USED BY THE GUI
    // The value the gui wants to set
    double gui_value() const;
    void set_value(double value);
    bool changed() const;



private:
    // The active value, used by the user, only one user means set will override its own local
    double value_;
    std::atomic<double> new_value;
    std::atomic<bool> current{true};
    std::mutex mtx;
    double validate(double a) const;
};

}

