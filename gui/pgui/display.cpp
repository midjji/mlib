#include <display.h>
#include <QGridLayout>
#include <QLabel>
#include <label.h>


QWidget* parameter_widget_int_range(IntRangeParameter* param){

}

QWidget* parameter_widget_float_range(IntRangeParameter* param){

}

QWidget* parameter_widget(Parameter* param)
{
    switch (param->type){
    case Parameter::type_t::integer_range:
        return parameter_widget_int_range();
    case Parameter::type_t::float_range:
        return
    }
}

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
    for(auto param:ps->ps) {
        layout->addWidget(parameter_widget(param));
    }

    // only add the params!




}



