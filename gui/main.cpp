#include <QApplication>
#include <editor.h>



int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ParameterEditor w;
    w.show();
    return a.exec();
}

