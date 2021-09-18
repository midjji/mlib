#include <QGridLayout>
#include <QLabel>

#include <display.h>
#include <label.h>
#include <pset.h>
#include <param_widget.h>

namespace cvl
{


PSetDisplayWidget::PSetDisplayWidget(
        PSetPtr ps,
        QWidget* parent):
    QWidget(parent)
{
    new QVBoxLayout(this);
    layout()->setMargin( 0 );
    layout()->setSpacing( 0 );
    setMinimumSize(200,640);
    QScrollArea *area = new QScrollArea(this);

    layout()->addWidget(area);
    area->setWidgetResizable(true);
    area->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    //layout()->setSizeConstraint(QLayout::SetMinimumSize);
    area->setSizeAdjustPolicy(QAbstractScrollArea::SizeAdjustPolicy::AdjustToContents);
    {
        QWidget* w=new QWidget(this);
        area->setWidget(w);



        // note that this also sets layout
        QVBoxLayout* layout=new QVBoxLayout(w);
        layout->setSizeConstraint(QLayout::SetMinimumSize);

        if(!ps) return; // no leak, this owns due to setting parent above
        layout->addWidget(new Label(ps->name, w),0);
        layout->addWidget(new Label(ps->desc, w),0);


        std::map<std::string,
                std::vector<Parameter*>> groups=ps->param_groups();
        for(auto& group:groups)
        {
            layout->addWidget(display_group(group.second, group.first,w),0);
        }
        layout->addStretch(1);

    }

    area->adjustSize();

}



}
