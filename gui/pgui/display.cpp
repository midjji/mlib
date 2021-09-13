#include <QGridLayout>
#include <QLabel>

#include <display.h>
#include <label.h>
#include <pset.h>
namespace cvl
{

ParamDisplayWidget::ParamDisplayWidget(
        ParamSet* ps,
        QWidget* parent):
    QWidget(parent)
{

    // note that this also sets layout
    QVBoxLayout* layout=new QVBoxLayout(this);
    if(!ps) return; // its wrong, qt takes care of it
    layout->addWidget(new Label(ps->name, this));
    layout->addWidget(new Label(ps->desc, this));
    //for(auto param:ps->ps) {
    //    layout->addWidget(parameter_widget(param));
    //}

    // only add the params!




}



}
