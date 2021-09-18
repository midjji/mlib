#pragma once
/* ********************************* FILE ************************************/
/** \file    database.h
 *
 * \brief    This header contains the annotation and results databases for the daimler dataset
 *
 * \remark
 * - c++11
 *
 * \todo
 *
 *
 *
 * \author   Mikael Persson
 * \date     2019-01-01
 * \note GPL licence
 *
 ******************************************************************************/

#include <mlib/datasets/daimler/results_types.h>
namespace cvl{
namespace mtable{



using result_db_type= decltype (
sqlite_orm::make_storage(
    "path",
    sqlite_orm::make_table(
        "imo_results",
        sqlite_orm::make_column("uid", &Imoresrow::uid,
                                sqlite_orm::autoincrement(),
                                sqlite_orm::primary_key()),
        sqlite_orm::make_column("frame_id", &Imoresrow::frame_id),
        sqlite_orm::make_column("imo_id", &Imoresrow::imo_id),
        sqlite_orm::make_column("pose0", &Imoresrow::pose0),
        sqlite_orm::make_column("pose1", &Imoresrow::pose1),
        sqlite_orm::make_column("pose2", &Imoresrow::pose2),
        sqlite_orm::make_column("pose3", &Imoresrow::pose3),
        sqlite_orm::make_column("pose4", &Imoresrow::pose4),
        sqlite_orm::make_column("pose5", &Imoresrow::pose5),
        sqlite_orm::make_column("pose6", &Imoresrow::pose6),

        sqlite_orm::make_column("fxm_x", &Imoresrow::fxm_x),
        sqlite_orm::make_column("fxm_y", &Imoresrow::fxm_y),
        sqlite_orm::make_column("fxm_z", &Imoresrow::fxm_z),

        sqlite_orm::make_column("xm_x", &Imoresrow::xm_x),
        sqlite_orm::make_column("xm_y", &Imoresrow::xm_y),
        sqlite_orm::make_column("xm_z", &Imoresrow::xm_z),


        sqlite_orm::make_column("row_start", &Imoresrow::row_start),
        sqlite_orm::make_column("col_start", &Imoresrow::col_start),
        sqlite_orm::make_column("row_end", &Imoresrow::row_end),
        sqlite_orm::make_column("col_end", &Imoresrow::col_end)
        ),
    sqlite_orm::make_table(
        "egomotion",
        sqlite_orm::make_column("uid", &Egomotionrow::uid, sqlite_orm::autoincrement(), sqlite_orm::primary_key()),
        sqlite_orm::make_column("frame_id", &Egomotionrow::frame_id),
        sqlite_orm::make_column("pose0", &Egomotionrow::pose0),
        sqlite_orm::make_column("pose1", &Egomotionrow::pose1),
        sqlite_orm::make_column("pose2", &Egomotionrow::pose2),
        sqlite_orm::make_column("pose3", &Egomotionrow::pose3),
        sqlite_orm::make_column("pose4", &Egomotionrow::pose4),
        sqlite_orm::make_column("pose5", &Egomotionrow::pose5),
        sqlite_orm::make_column("pose6", &Egomotionrow::pose6)),
    sqlite_orm::make_table(
        "timing",
        sqlite_orm::make_column("uid", &Timingrow::uid, sqlite_orm::autoincrement(), sqlite_orm::primary_key()),
        sqlite_orm::make_column("frame_id", &Timingrow::frame_id),
        sqlite_orm::make_column("time_ns", &Timingrow::time_ns)),
    sqlite_orm::make_table(
        "velocity",
        sqlite_orm::make_column("uid", &Vdata::uid,
                                sqlite_orm::autoincrement(),
                                sqlite_orm::primary_key()),
        sqlite_orm::make_column("frameid", &Vdata::frameid),
        sqlite_orm::make_column("imoid", &Vdata::imoid),
        sqlite_orm::make_column("imo_vx", &Vdata::imo_vx),
        sqlite_orm::make_column("imo_vy", &Vdata::imo_vy),
        sqlite_orm::make_column("imo_vz", &Vdata::imo_vz),
        sqlite_orm::make_column("ekf_vx", &Vdata::ekf_vx),
        sqlite_orm::make_column("ekf_vy", &Vdata::ekf_vy),
        sqlite_orm::make_column("ekf_vz", &Vdata::ekf_vz))
    ));

struct GTRow{
    int uid;
    int frame_id;
    int imo_id;
    float row_start, col_start, row_end, col_end;
    float x,y,z;
    Vector3d xm() const{return Vector3d(x,y,z);}
    void set_xm(Vector3d xm){x=float(xm[0]);y=float(xm[1]);z=float(xm[2]);}
    BoundingBox bb() const{return BoundingBox(imo_id, row_start, col_start, row_end, col_end);}
    double area() const{return bb().area();}
};
template <class Struct, class Element> using sqct = sqlite_orm::internal::column_t<Struct, Element, const Element& (Struct::*)() const, void (Struct::*)(Element)>;
using gt_db_type=sqlite_orm::internal::storage_t<
sqlite_orm::internal::table_t<GTRow,
sqlite_orm::internal::column_t<GTRow, int, const int& (GTRow::*)() const,
void (GTRow::*)(int), sqlite_orm::constraints::autoincrement_t, sqlite_orm::constraints::primary_key_t<> >,
sqct<GTRow, int>, sqct<GTRow, int>,
sqct<GTRow, float>, sqct<GTRow, float>, sqct<GTRow, float>, sqct<GTRow, float>,
sqct<GTRow, float>, sqct<GTRow, float>, sqct<GTRow, float> > >;


struct GTDB{
    mtable::gt_db_type db;
    GTDB(std::string path):db(sqlite_orm::make_storage(path.c_str(),
                                                 sqlite_orm::make_table("boundingboxes",
                                                           sqlite_orm::make_column("id", &mtable::GTRow::uid, sqlite_orm::autoincrement(), sqlite_orm::primary_key()),
                                                           sqlite_orm::make_column("frame_id", &mtable::GTRow::frame_id),
                                                           sqlite_orm::make_column("imo_id", &mtable::GTRow::imo_id),
                                                           sqlite_orm::make_column("row_start", &mtable::GTRow::row_start),
                                                           sqlite_orm::make_column("col_start", &mtable::GTRow::col_start),
                                                           sqlite_orm::make_column("row_end", &mtable::GTRow::row_end),
                                                           sqlite_orm::make_column("col_end", &mtable::GTRow::col_end),
                                                           sqlite_orm::make_column("x", &mtable::GTRow::x),
                                                           sqlite_orm::make_column("y", &mtable::GTRow::y),
                                                           sqlite_orm::make_column("z", &mtable::GTRow::z)))){};
};


}

}
