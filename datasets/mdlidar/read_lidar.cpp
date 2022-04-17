#include <fstream>
#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>
#include <iostream>

#include "read_lidar.h"

// protobufs
#include "mdlidar.pb.h"


using namespace std;
namespace cvl {
namespace md {



struct LaserCalib {
    float horiz_offset;
    float vert_offset;
    float rot_correction;
    float vert_correction;
};
std::vector<LaserCalib> read_calibration()
{



    std::string calib_file_path = "/storage/datasets/mdlidar/vel_64_calib.csv";
    //cout<<"parsing calib file: \""<<calib_file_path<<"\n"<<endl;
    std::vector<LaserCalib> calib;
    std::ifstream in(calib_file_path);
    std::string line;

    while (std::getline(in, line)) {
        std::stringstream sep(line);
        std::string field;
        LaserCalib temp;

        std::getline(sep, field, ',');
        temp.horiz_offset = std::stof(field);
        std::getline(sep, field, ',');
        temp.vert_offset = std::stof(field);
        std::getline(sep, field, ',');
        temp.rot_correction = std::stof(field);
        std::getline(sep, field, ',');
        temp.vert_correction = std::stof(field);

        calib.push_back(temp);
    }
    return calib;
}


std::vector<md::LidarObservation> read_lidar_frame(int data_id){
    // Parse calibration file
    auto calib=read_calibration();

    std::string base_file_path = "/storage/datasets/mdlidar/town_1/";
    //cout<<"parsing: "<<base_file_path<<endl;
    std::vector<md::LidarObservation> obs; obs.reserve(1e6);

    mdlidar::Header header;
    {
        // Deserialize header data file
        std::fstream input2(base_file_path + "header",
                            std::ios::in | std::ios::binary);

        header.ParseFromIstream(&input2);
    }

    // Deserialize single data file



    std::string path=base_file_path + "frame/data_" + std::to_string(data_id);
    //cout<<"parsing frame path: "<<path<<endl;
    std::fstream input(path,
                       std::ios::in | std::ios::binary);
    mdlidar::Frame data;
    data.ParseFromIstream(&input);



    // Process raw lidar data to a local pointcloud
    uint points_per_channel = header.points_count_by_channel();
    int64_t start_time = data.start_time();
    int64_t end_time = data.end_time();



    for (int i = 0; i < data.points_size(); ++i)
    {
        float range = data.points(i).range();
        float rotation = data.points(i).rotation()*M_PI/180.0;  // convert to rad

        // Ignore -1 laser returns
        if (range > 0)
        {
            int laser_id = i/points_per_channel;
            float cos_vert_angle = cos(calib[laser_id].vert_correction);
            float sin_vert_angle = sin(calib[laser_id].vert_correction);
            float cos_rot_correction = cos(calib[laser_id].rot_correction);
            float sin_rot_correction = sin(calib[laser_id].rot_correction);

            float cos_rot_angle =
                    cos(rotation)*cos_rot_correction + sin(rotation)*sin_rot_correction;
            float sin_rot_angle =
                    sin(rotation)*cos_rot_correction - cos(rotation)*sin_rot_correction;

            float xy_distance =
                    range*cos_vert_angle - calib[laser_id].vert_offset*sin_vert_angle;

            // Point coordinates in meters
            float x = xy_distance*cos_rot_angle + calib[laser_id].horiz_offset*sin_rot_angle;
            float y = -xy_distance*sin_rot_angle + calib[laser_id].horiz_offset*cos_rot_angle;
            float z = range*sin_vert_angle + calib[laser_id].vert_offset*cos_vert_angle;

            // Time in nanoseconds
            int64_t time =
                    start_time + double(i%points_per_channel)/(points_per_channel - 1)
                    *(end_time - start_time);
            obs.push_back(md::LidarObservation(time, x,y,z, data_id));
        }
    }

    cout<<"found "<<obs.size()<<" lidar points "<<endl;
    return obs;
}

std::vector<md::LidarObservation> read_lidar()
{

    // Parse calibration file
    auto calib=read_calibration();

    std::string base_file_path = "/storage/datasets/mdlidar/town_1/";
    // cout<<"parsing: "<<base_file_path<<endl;
    std::vector<md::LidarObservation> obs; obs.reserve(1e6);

    mdlidar::Header header;
    {
        // Deserialize header data file
        std::fstream input2(base_file_path + "header",
                            std::ios::in | std::ios::binary);

        header.ParseFromIstream(&input2);
    }

    // Deserialize single data file
    for(int data_id=0;data_id<3000;++data_id)
    {


        std::string path=base_file_path + "frame/data_" + std::to_string(data_id);
        // cout<<"parsing frame path: "<<path<<endl;
        std::fstream input(path,
                           std::ios::in | std::ios::binary);
        mdlidar::Frame data;
        data.ParseFromIstream(&input);



        // Process raw lidar data to a local pointcloud
        uint points_per_channel = header.points_count_by_channel();
        int64_t start_time = data.start_time();
        int64_t end_time = data.end_time();



        for (int i = 0; i < data.points_size(); ++i)
        {
            float range = data.points(i).range();
            float rotation = data.points(i).rotation()*M_PI/180.0;  // convert to rad

            // Ignore -1 laser returns
            if (range > 0)
            {
                int laser_id = i/points_per_channel;
                float cos_vert_angle = cos(calib[laser_id].vert_correction);
                float sin_vert_angle = sin(calib[laser_id].vert_correction);
                float cos_rot_correction = cos(calib[laser_id].rot_correction);
                float sin_rot_correction = sin(calib[laser_id].rot_correction);

                float cos_rot_angle =
                        cos(rotation)*cos_rot_correction + sin(rotation)*sin_rot_correction;
                float sin_rot_angle =
                        sin(rotation)*cos_rot_correction - cos(rotation)*sin_rot_correction;

                float xy_distance =
                        range*cos_vert_angle - calib[laser_id].vert_offset*sin_vert_angle;

                // Point coordinates in meters
                float x = xy_distance*cos_rot_angle + calib[laser_id].horiz_offset*sin_rot_angle;
                float y = -xy_distance*sin_rot_angle + calib[laser_id].horiz_offset*cos_rot_angle;
                float z = range*sin_vert_angle + calib[laser_id].vert_offset*cos_vert_angle;

                // Time in nanoseconds
                int64_t time =
                        start_time + double(i%points_per_channel)/(points_per_channel - 1)
                        *(end_time - start_time);
                obs.push_back(md::LidarObservation(time, x,y,z, data_id));
            }
        }
    }
    //cout<<"found "<<obs.size()<<" lidar points "<<endl;
    return obs;
}



template<class Orientation, class Position>
PoseD pose(const Orientation& o, const Position& p)
{
    // t in world coordinates...
    float sensor_position_x = p.x();
    float sensor_position_y = p.y();
    float sensor_position_z = p.z();
    Vector3d T_world(sensor_position_x ,sensor_position_y, sensor_position_z);

    // angle axis, Rvi
    float sensor_orientation_x = o.axis().x();
    float sensor_orientation_y = o.axis().y();
    float sensor_orientation_z = o.axis().z();
    float sensor_orientation_angle = o.angle();

    Vector3d n(sensor_orientation_x,sensor_orientation_y, sensor_orientation_z);
    //cout<<"axis"<<n<< " "<<n.norm()<<endl;
    if(n.norm()>1e-6) n.normalize();
    //cout<<" angle: "<<sensor_orientation_angle<<endl;
    Vector4d uq(std::cos(0.5*sensor_orientation_angle),
                std::sin(0.5*sensor_orientation_angle)*sensor_orientation_x,
                std::sin(0.5*sensor_orientation_angle)*sensor_orientation_y,
                std::sin(0.5*sensor_orientation_angle)*sensor_orientation_z);
    //cout<<"uq"<<uq<< " "<<uq.norm()<<endl;
    uq.normalize();
    PoseD P(uq,T_world);
    P.setT(P.getTinW()); // is this thing inverted or not? docs say yes, looks like NO
    return P;
}


std::map<long double, PoseD> read_lidar_gt_poses()
{
    std::map<long double, PoseD> poses;

    PoseD Pv0;

    // Deserialize single data file
    for(int data_id = 0;data_id<3000;++data_id)
    {
        mdlidar::Frame data;
        std::string base_file_path = "/storage/datasets/mdlidar/town_1/";
        std::string path=base_file_path + "frame/data_" + std::to_string(data_id);
        //cout<<"parsing frame path: "<<path<<endl;
        std::fstream input(path,
                           std::ios::in | std::ios::binary);
        data.ParseFromIstream(&input);

        // Sensor vehicle state, the current one, they also give the preceeding one at t-1e-5, though no real need for it, save better gt...

        Pose Pvw=pose(data.state().orientation(),data.state().position());
        long double time=data.start_time();
        time*=1e-9;

        poses[time]=Pvw;
        //cout<<"time:" <<time<<endl;
    }
    return poses;
}

std::map<int, PoseD> read_lidar_gt_poses_by_frame()
{
    std::map<int, PoseD> poses;

    // Deserialize single data file
    for(int data_id = 0;data_id<3000;++data_id)
    {
        mdlidar::Frame data;
        std::string base_file_path = "/storage/datasets/mdlidar/town_1/";
        std::string path=base_file_path + "frame/data_" + std::to_string(data_id);
        //cout<<"parsing frame path: "<<path<<endl;
        std::fstream input(path,
                           std::ios::in | std::ios::binary);
        data.ParseFromIstream(&input);

        // Sensor vehicle state, the current one, they also give the preceeding one at t-1e-5, though no real need for it, save better gt...

        poses[data_id]=pose(data.state().orientation(),data.state().position());
        //cout<<"time:" <<time<<endl;
    }
    return poses;
}
}
}
std::string str(cvl::md::LidarObservation ob){
    std::stringstream ss;
    ss<<ob.frame<<", "<<ob.time<< ": "<<ob.xs[0]<<", "<<ob.xs[1]<<", "<<ob.xs[2];
    return ss.str();
}
