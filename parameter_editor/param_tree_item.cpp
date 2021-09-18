#include <param_tree_item.h>
#include <pset.h>
namespace cvl {
ParamItem::ParamItem(PSetPtr ps):
    QStandardItem(ps->name.c_str()), ps(ps){
    setEditable(false);
}
}
