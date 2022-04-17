#include <fstream>
#include <iostream>
#include "mdlidar.pb.h"

int main(int argc, char *argv[]){
	
  // Deserialize single data file
  int data_id = 7;  // file to deserialize is data_7 (arbitrary example)
  mdlidar::Frame data;
  std::fstream input("/path/to/frame/data_" 
      + std::to_string(data_id), std::ios::in | std::ios::binary);
  data.ParseFromIstream(&input);

  // Sensor vehicle state
  float sensor_position_x = data.state().position().x();
  float sensor_position_y = data.state().position().y();
  float sensor_position_z = data.state().position().z();
  float sensor_orientation_x = data.state().orientation().axis().x();
  float sensor_orientation_y = data.state().orientation().axis().y();
  float sensor_orientation_z = data.state().orientation().axis().z();
  float sensor_orientation_angle = data.state().orientation().angle();
  
  // Object state (repeated field)
  for (int i = 0; i < data.object_state_size(); ++i) {
    float position_x = data.object_state(i).position().x();
    float position_y = data.object_state(i).position().x();
    float position_z = data.object_state(i).position().x();
    float orientation_x = data.object_state(i).orientation().axis().x();
    float orientation_y = data.object_state(i).orientation().axis().y();
    float orientation_z = data.object_state(i).orientation().axis().z();
    float orientation_angle = data.object_state(i).orientation().angle();
  } // end loop i

  return 0;
}