#pragma once
#include "parameter.h"
#include <atomic>
#include <mutex>
#include <map>
namespace cvl {



struct EnumParameter:public Parameter {
    EnumParameter(int default_,
            std::map<int, std::string> options,
            std::string name="unnamed",
            std::string group="",
            std::string desc="no tool tip");

    // the currently cached selected value,
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
    const std::map<int, std::string> options;
    // The active value, used by the user, update_value changes local
    int value_;
    std::atomic<int> new_value;
    std::atomic<bool> current{true};
    int validate(int a) const;
};
}
