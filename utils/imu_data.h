#pragma once
#include <mlib/utils/cvl/matrix.h>
#include <mlib/utils/mlibtime.h>

namespace cvl {

namespace imu {


class Data
{
public:
    enum {invalid=0, gyro_valid=1, acc_valid=2};

    float128 time; // in seconds
    Vector3d acc;// in m/s²
    Vector3d gyro;// in rads/s
    int state;

    Data()=default;
    Data(float128 time,
         cvl::Vector3d acc,
         cvl::Vector3d gyro,
         int state=gyro_valid|acc_valid):
        time(time),acc(acc),gyro(gyro),state(state){

    }
};
}
}
