#if 0
#pragma once

#include <memory>
#include <vector>
#include <mlib/vis/flow.h>
#include <mlib/vis/flow_field.h>

namespace cvl {

void show_flow(std::shared_ptr<vis::FlowField> flows, std::string name="");
void show_flow(std::vector<Flow> flows, std::string name="");

bool flow_field_viewers_open();
void show_trajectory(std::vector<PoseD> ps, std::string name="");
}
#endif
