#include "param_tree.h"
#include <param_tree_item.h>
#include <pset.h>

#include <QTreeView>
#include <QStandardItemModel>
#include <QItemSelectionModel>
#include <QStandardItem>
#include <QTreeView>

#include <QGridLayout>
#include <QLabel>
#include <QMenuBar>
#include <QStatusBar>

#include <QHeaderView>
#include <iostream>


using std::cout;
using std::endl;

namespace cvl {




ParamTree::ParamTree(QWidget* parent):
    QWidget(parent),
    tree(new QTreeView(this))
{    
    setMinimumSize(200,640);
    // sets item model
    tree->setModel(new QStandardItemModel(tree));
    tree->expandAll();
    tree->setHeaderHidden(true);





    //selection changes shall trigger a slot
#if 0
    connect(tree->selectionModel(),
            &QItemSelectionModel::selectionChanged,
            this,
            [&](){do});
#endif

}
QStandardItemModel* ParamTree::item_model() {
    return (QStandardItemModel*)(tree->model());
}

ParamItem* ParamTree::selected_item() {
    QModelIndex index = tree->currentIndex();
    if(!index.isValid()) return nullptr; // if the user has selected nothing
    return (ParamItem*)(item_model()->itemFromIndex(index));
}
void ParamTree::clear()
{
    QStandardItem *root = item_model()->invisibleRootItem();
    root->clearData();
}


void addtree(PSetPtr ps,
             QStandardItem * node){
    auto pi=new ParamItem(ps);
    node->appendRow(pi);
    for(const auto& p:ps->subsets())
        addtree(p,pi);
}
void ParamTree::add(PSetPtr ps)
{
    addtree(ps, item_model()->invisibleRootItem());
}

}
