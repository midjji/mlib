#pragma once
#include <memory>
#include <QStandardItem>

namespace cvl {
class PSet;
class ParamItem:public QStandardItem{
public:
    ParamItem(std::shared_ptr<PSet> ps);
    std::shared_ptr<PSet> ps;
};
}
