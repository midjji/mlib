#include <label.h>


Label::Label(std::string name,
             QWidget* parent):QLabel(parent){
    setText(name.c_str());
}
