#pragma once
#include <QLabel>
namespace cvl {


class Label:public QLabel{
public:
    Label(std::string name,
          QWidget* parent=0);
};

}
