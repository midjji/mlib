#pragma once
#include <QLabel>
namespace cvl {


class Label:public QLabel{
public:
    Label(std::string name="unset label",
          QWidget* parent=0);
    void set_text(std::string str);
    void set_value(int v);
    void set_value(float v);
    void set_value(double v);
    void set_value(long double v);
};

}
