#pragma once

#include <QWidget>
#include <memory>

class QTreeView;
class QStandardItemModel;

namespace cvl {



class ParamItem;
class ParamSet;

class ParamTree:public QWidget
{
    Q_OBJECT
public:
    QTreeView* tree;
    QStandardItemModel* item_model();
    ParamItem* selected_item();
    ParamTree(QWidget* parent = nullptr);
    void clear();
    void add(std::shared_ptr<ParamSet> ps);

};

}
