#include <display.h>
#include <QGridLayout>
#include <QLabel>
#include <label.h>
ParamDisplayWidget::ParamDisplayWidget(
        ParamSet* ps,
        QWidget* parent):
    QWidget(parent)
{

    // note that this also sets layout
    QVBoxLayout* layout=new QVBoxLayout(this);
    if(!ps) return;
    layout->addWidget(new Label(ps->name, this),0,0);
    layout->addWidget(new Label(ps->desc, this),0,1);
    // only add the params!




}



