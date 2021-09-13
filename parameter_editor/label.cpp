#include <label.h>

namespace cvl {


Label::Label(std::string name,
             QWidget* parent):QLabel(parent){
    setText(name.c_str());

}
}
