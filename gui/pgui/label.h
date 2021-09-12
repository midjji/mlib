#pragma once
#include <QLabel>
class Label:public QLabel{
public:
    Label(std::string name,
          QWidget* parent=0);
};

