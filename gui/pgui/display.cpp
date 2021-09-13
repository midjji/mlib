#include <QGridLayout>
#include <QLabel>

#include <display.h>
#include <label.h>
#include <pset.h>
#include <param_widget.h>

namespace cvl
{
ParamSetDisplayWidget::ParamSetDisplayWidget(
        ParamSetPtr ps,
        QWidget* parent):
    QWidget(parent)
{

    // note that this also sets layout
    QVBoxLayout* layout=new QVBoxLayout(this);
    if(!ps) return; // no leak, this owns due to setting parent above
    layout->addWidget(new Label(ps->name, this));
    layout->addWidget(new Label(ps->desc, this));

    std::map<std::string,
            std::vector<Parameter*>> groups=ps->param_groups();
    for(auto& group:groups)
    {
        layout->addWidget(display_group(group.second, group.first,this));
    }
}



}
