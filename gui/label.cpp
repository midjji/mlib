#include <mlib/gui/label.h>
#include <sstream>
namespace cvl {


Label::Label(std::string name,
             QWidget* parent):QLabel(parent)
{

    setText(QString::fromStdString(name));

}
void Label::set_text(std::string str){
    setText(QString::fromStdString(str));
}

void Label::set_value(int v){
    std::stringstream ss;
    ss<<v;
    set_text(ss.str());
}
void Label::set_value(float v){
    std::stringstream ss;
    ss<<v;
    set_text(ss.str());
}
void Label::set_value(double v){
    std::stringstream ss;
    ss<<v;
    set_text(ss.str());
}
void Label::set_value(long double v){
    std::stringstream ss;
    ss<<v;
    set_text(ss.str());
}

}
