#pragma once
#include <sstream>
#include <vector>
#include <mlib/utils/cvl/matrix.h>

namespace cvl{

class BoundingBox{
public:
    int id;
    double row_start=0;
            double col_start=0;
    double row_end=0;
    double col_end=0;
    BoundingBox()=default;
    BoundingBox(int id, double row_start, double col_start, double row_end, double col_end);
    bool in(double row, double col) const;
    bool in(Vector2d rc) const;
    double area() const;
    double rows() const ;
    double cols() const ;
    std::vector<Vector2d> corners() const;
    bool intersects(const BoundingBox& bb) const;
    BoundingBox intersect(const BoundingBox& bb) const;
    double iou (const BoundingBox& bb) const;
    BoundingBox remove_border(double margin);
    bool near_image_edge(double margin, double rows, double cols) const;
    void include(Vector2d y);
};
} // end namespace cvl
std::istream& operator>>(std::istream& is, cvl::BoundingBox& bb);
std::ostream& operator<<(std::ostream& os, cvl::BoundingBox bb);
