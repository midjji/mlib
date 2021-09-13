#pragma once
#include "parameter.h"
#include <atomic>
#include <mutex>
namespace cvl {



struct IntParameter:public Parameter {
    IntParameter(
            int default_value,
            std::string name="unnamed",
            std::string group="",
            std::string desc="no tool tip",
            int minv=std::numeric_limits<int>::min(),
            int maxv=std::numeric_limits<int>::max());
    // the value range
    const int minv;
    const int maxv;
    bool ranged() const;
    // USED BY THE Parametrized class...
    // the user value
    int value() const;
    // the user selects when to update
    bool update_value() override;
    type_t type() const override;

    //USED BY THE GUI
    // The value the gui wants to set
    int gui_value() const;
    void set_value(int value);
    bool changed() const;


private:
    // The active value, used by the user, update_value changes local
    int value_;
    std::atomic<int> new_value;
    std::atomic<bool> current{true};
    std::mutex mtx;
    int validate(int a) const;
};
}
