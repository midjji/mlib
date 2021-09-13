#pragma once
#include <QMainWindow>
#include <memory>
namespace cvl {

class ParamSet;
class ParamWidget;
/**
 * @brief The ParameterEditor class
 * QMainWindow provides ready layout for
 *  - a menubar (top),
 *  - a statusbar (bottom)
 *  - docking around the center)
 *  - a central widget of your chosing
 *
 *  Only use this if you want standard looking file menu etc...
 *  Otherwize there is no point
 *
 */
class ParameterEditor: public QMainWindow
{
    Q_OBJECT
private:
    // does not need to be a member unless you will add or remove stuff
    ParamWidget* central;

public:
    ParameterEditor(QWidget *parent = 0);
    void set(std::shared_ptr<ParamSet> p);
};
}
