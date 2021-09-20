#include <QFileDialog>

#include <mlib/gui/filedialog.h>


std::string get_path_from_user()
{
    QString start_dir{QDir::homePath()};

    QString path;
    while (path.isEmpty()) {
        path = QFileDialog::getExistingDirectory(
                    nullptr, "Select logging root directory", start_dir,
                    QFileDialog::ShowDirsOnly | QFileDialog::DontUseNativeDialog);
    }

    return path.toStdString();
}
