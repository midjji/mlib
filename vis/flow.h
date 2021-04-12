#pragma once
#include <mlib/utils/cvl/pose.h>
namespace cvl {
struct Flow{
    Vector3d origin,velocity,color;
    Flow()=default;
    Flow(Vector3d origin,
         Vector3d velocity,
         Vector3d color);

    void apply_transform(PoseD pose);
    bool is_normal() const;
};
}
