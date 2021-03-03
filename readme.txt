Under GPL

# install the required dependencies
# I think these are sufficient, but if not the list below is!
# hard dependencies ! 
ceres-solver eigen3 


sudo apt install build-essential cmake libceres-dev curl libsqlite3-dev git libqt5svg5-dev libeigen3-dev  ccache libtiff5-dev libpng-dev libopenexr-dev libwebp-dev libopenblas-dev
#
#sudo apt install screen cuda meld cppcheck svn ccache clang-10 evince gimp scons libpython3.8-dev openssl software-properties-common qt5-default libqt5svg5-dev texstudio texlive-full qtcreator vlc chromium-browser libceres-dev git curl cmake cmake-curses-gui build-essential gparted libeigen3-dev openssh-server ufw ccache libtiff5-dev libpng-dev libopenexr-dev libwebp-dev libopenblas-dev liblapack-dev libgtkglext1-dev libgtkglextmm-x11-1.2-dev libxml2-dev libcurl4-openssl-dev libsvgpp-dev libsqlite3-dev gparted  git build-essential cmake python3-dev python3-setuptools python3-pytest python3-scipy python3-h5py libceres-dev libpython-dev
# if the build does not work as it should, try installing cmake from git
# for mlib-cuda install cmake from git
#https://github.com/Kitware/CMake

# install ceres opencv and openscenegraph from git
https://github.com/ceres-solver/ceres-solver.git
https://github.com/opencv/opencv.git
https://github.com/openscenegraph/OpenSceneGraph.git
# go to mlib/apps/mlib/extern and run the clone-dependencies.sh
then go to mlib/apps/mlib/build and run
cmake ..
# look for any errors, or missing dependencies, then 
make -j8










# this is a incremental tested refactor of mlib
# the goals are 
# 1) unit tests
# 2) code coaleasing
# 3) documentation
# 4) simplified reuse
# 5) anti necromancy efforts
