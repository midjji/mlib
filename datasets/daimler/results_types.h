#pragma once
#include <mlib/utils/cvl/pose.h>
#include <mlib/utils/bounding_box.h>
#include <sqlite_orm.h>
namespace cvl
{
struct Timingrow
{
    int uid;
    int frame_id;
    int time_ns;
    Timingrow(int uid, int frame_id, uint64_t time_ns):uid(uid),frame_id(frame_id),time_ns(time_ns){}
};
struct Egomotionrow
{
    Egomotionrow()=default;
    Egomotionrow(int uid, int frame_id, PoseD pose):uid(uid),frame_id(frame_id),
        pose0(pose[0]),
        pose1(pose[1]),
        pose2(pose[2]),
        pose3(pose[3]),
        pose4(pose[4]),
        pose5(pose[5]),
        pose6(pose[6]){}
    int uid;
    int frame_id;
    double pose0, pose1, pose2, pose3, pose4, pose5, pose6; // q, t; Pcw(from frame_id 0)
    PoseD pose() const{return PoseD(Vector4d(pose0,pose1,pose2,pose3), Vector3d(pose4,pose5,pose6));}

};
struct Vdata{
    Vdata()=default;
    Vdata(int age, uint frameid, int imoid, Vector3d imo_v,Vector3d ekf_v)
        :age(age),frameid(frameid),imoid(imoid),
          imo_vx(imo_v[0]),imo_vy(imo_v[1]),imo_vz(imo_v[2]),
    ekf_vx(ekf_v[0]),ekf_vy(ekf_v[1]),ekf_vz(ekf_v[2]){}

    int uid;
    int age;
    uint frameid;
    int imoid;
    double imo_vx,imo_vy,imo_vz;
    double ekf_vx,ekf_vy,ekf_vz;

};



struct Imoresrow
{
    int uid;
    int frame_id;
    int imo_id;
    double pose0, pose1, pose2, pose3, pose4, pose5, pose6; // q, t;

    double  fxm_x,fxm_y,fxm_z; // using one
    double  xm_x,xm_y,xm_z; // using all,
    double row_start, col_start, row_end, col_end;


    Imoresrow()=default;
    Imoresrow(int uid, int frame_id, int imo_id,
              PoseD pose, // Pc_imo, specifically to this frame_id camera.
              Vector3d fxm, // in imo coordinates!
              Vector3d xm,  // in imo coordinates!
              BoundingBox bb):
        uid(uid), frame_id(frame_id), imo_id(imo_id),
        pose0(pose[0]),
        pose1(pose[1]),
        pose2(pose[2]),
        pose3(pose[3]),
        pose4(pose[4]),
        pose5(pose[5]),
        pose6(pose[6]),
        fxm_x(fxm[0]),
        fxm_y(fxm[1]),
        fxm_z(fxm[2]),
        xm_x(xm[0]),
        xm_y(xm[1]),
        xm_z(xm[2]),
        row_start(bb.row_start),
        col_start(bb.col_start),
        row_end(bb.row_end),
        col_end(bb.col_end){}
    PoseD pose() const{return PoseD(Vector4d(pose0,pose1,pose2,pose3), Vector3d(pose4,pose5,pose6));}
    Vector3d fxm_in_camera() const{return pose()*Vector3d(fxm_x,fxm_y,fxm_z);}
    Vector3d xm_in_camera() const{return pose()*Vector3d(xm_x,xm_y,xm_z);}
    BoundingBox bb() const{return BoundingBox(imo_id, row_start, col_start, row_end, col_end);}
};
inline bool operator< (Imoresrow a, Imoresrow b){return a.uid<b.uid;}

struct imoframeres
{
    int frame_id;
    PoseD Pci;// P_{c,imo}
    Vector3d fxm; // median x of this frameid
    BoundingBox bb;
    imoframeres(int frame_id, PoseD Pci, Vector3d fxm, BoundingBox bb):frame_id(frame_id), Pci(Pci), fxm(fxm), bb(bb){}
};

struct imores{


    int imo_id;
    Vector3d xm; // median x of all features, in imo
    std::vector<imoframeres> res;
    std::vector<Imoresrow> resrows(){
        std::vector<Imoresrow> rets;rets.reserve(res.size());
        for(auto ifr:res)
            rets.push_back(Imoresrow(-1,ifr.frame_id, imo_id, ifr.Pci, ifr.fxm, xm, ifr.bb ));
        return rets;
    }

};

}
