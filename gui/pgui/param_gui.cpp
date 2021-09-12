#include "param_gui.h"

#include <QTreeView>
#include <QStandardItemModel>
#include <QItemSelectionModel>

#include <QTreeView>
#include <QStandardItemModel>
#include <QItemSelectionModel>
#include <QGridLayout>
#include <QLabel>
#include <QMenuBar>
#include <QStatusBar>

#include <QHeaderView>
#include <iostream>
using std::cout;
using std::endl;

ParamItem::ParamItem(ParamSet* ps):
    QStandardItem(ps->name.c_str()), ps(ps){
    setEditable(false);
}

ParamTree::ParamTree(QWidget* parent):
    QWidget(parent),
    tree(new QTreeView(this))
{
    // sets item model
    tree->setModel(new QStandardItemModel(tree));




    //selection changes shall trigger a slot

    connect(tree->selectionModel(),
            &QItemSelectionModel::selectionChanged,
            this,
            &ParamTree::selected);

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


void addtree(ParamSet* ps,
             QStandardItem * node){
    auto pi=new ParamItem(ps);
    node->appendRow(pi);
    for(auto p:ps->pss)
        addtree(p,pi);
}
void ParamTree::add(ParamSet* ps)
{
    addtree(ps, item_model()->invisibleRootItem());




    //treeView->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
    //treeView->header()->setStretchLastSection(false);
    //treeView->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
    //treeView->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

}



void ParamTree::selected(
//        [[maybe_unused]] const QItemSelection & curr/*newSelection*/,
  //      [[maybe_unused]] const QItemSelection & prev/*oldSelection*/
)
{
    auto* item=selected_item();
    if(!item) return;
    cout<<"pi->ps->name: "<<item->ps->name<<endl;


}


