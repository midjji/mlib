#pragma once
#include <memory>
#include <QStandardItem>

namespace cvl {
class ParamSet;
class ParamItem:public QStandardItem{
public:
    ParamItem(std::shared_ptr<ParamSet> ps);
    std::shared_ptr<ParamSet> ps;
};
}
