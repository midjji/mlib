

git clone https://github.com/onqtam/doctest.git
git clone https://github.com/fnc12/sqlite_orm.git
git clone https://github.com/ceres-solver/ceres-solver.git ceres/ceres-solver


if [ "$(whoami)" == "mikael" ]; then
git clone git@github.com:midjji/JKQtPlotter.git
git clone git@github.com:midjji/convenient_multithreaded_qt_gui.git
git clone git@github.com:midjji/cmake.git
else
git clone https://github.com/midjji/JKQtPlotter.git
git clone https://github.com/midjji/convenient_multithreaded_qt_gui.git
git clone https://github.com/midjji/cmake.git
fi
