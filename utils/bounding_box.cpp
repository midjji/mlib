#include <mlib/utils/bounding_box.h>
std::istream& operator>>(std::istream& is, cvl::BoundingBox& bb){
    int id;
    double rs,cs,re,ce;

    is>>id;
    is>>rs;
    is>>cs;
    is>>re;
    is>>ce;
    // sanity check?
    bb=cvl::BoundingBox(id,rs,cs,re,ce);
    return is;
}
std::ostream& operator<<(std::ostream& os, cvl::BoundingBox bb){
    os<<bb.id<<" ";
    os<<bb.row_start<<" ";
    os<<bb.col_start<<" ";
    os<<bb.row_end<<" ";
    os<<bb.col_end<<" ";
    // sanity check?
    return os;
}

namespace cvl{

BoundingBox::BoundingBox(int id, double row_start, double col_start, double row_end, double col_end):
    id(id),row_start(row_start), col_start(col_start), row_end(row_end), col_end(col_end){
    if(row_start<row_end)
        std::swap(row_start, row_end);
    if(col_start<col_end)
        std::swap(col_start, col_end);
}

bool BoundingBox::in(double row, double col) const{
    return (row_start <= row && row<= row_end && col_start<=col && col<=col_end);
}
bool BoundingBox::in(Vector2d rc) const{
    return in(rc[0],rc[1]);
}
double BoundingBox::area() const{
    return (row_end - row_start)*(col_end - col_start);
}
double BoundingBox::rows() const {return  row_end - row_start;}
double BoundingBox::cols() const {return  col_end - col_start;}
std::vector<Vector2d> BoundingBox::corners() const{
    return {{row_start, col_start},
        {row_start, col_end},
        {row_end, col_start},
        {row_end, col_end},
    };
}
bool BoundingBox::intersects(const BoundingBox& bb) const{
    for(auto corner:corners())
        if(bb.in(corner))
            return true;
    for(auto corner:bb.corners())
        if(in(corner))
            return true;
    return false;
}
BoundingBox BoundingBox::intersect(const BoundingBox& bb) const{
    // intersection is always square! this is only correct if intersection exists, check first!
    if(intersects(bb))
        return BoundingBox(id, std::max(row_start, bb.row_start),
                           std::max(col_start, bb.col_start),
                           std::min(row_end, bb.row_end),
                           std::min(col_end, bb.col_end));
    return BoundingBox(-1,0,0,0,0);
}
double BoundingBox::iou (const BoundingBox& bb) const{
    if(!intersects(bb)) return -1;
    double i=intersect(bb).area();
    double union_area= area() + bb.area() - intersect(bb).area();

    if(i==0) return -1;
    if(union_area==0) return -1;
    return i/union_area;
}
BoundingBox BoundingBox::remove_border(double b){
    if((row_start + b<row_end -b) &&
            (col_start + b < col_end -b))
        return BoundingBox(id, row_start + b, col_start+b, row_end -b, col_end -b);
    return BoundingBox(-1,0,0,0,0);
}
bool BoundingBox::near_image_edge(double d, double rows, double cols) const {
    if (row_start -d <0) return true;
    if (col_start - d<0) return true;
    if(row_end +d > rows) return true;
    if(col_end + d> cols) return true;
    return false;
}
void BoundingBox::include(Vector2d y){
    double row=y[0];
    double col=y[1];
    if(row_start>row) row_start=row;
    if(col_start>col) col_start=col;
    if(row_end<row) row_end=row;
    if(col_end<col) col_end=col;
}
}
