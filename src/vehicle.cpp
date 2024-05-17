#include <vector>
#include <spdlog/spdlog.h>
#include <matplotlib-cpp/matplotlibcpp.h>
#include <fmt/core.h>
#include "vehicle.hpp"

namespace plt = matplotlibcpp;

Vehicle::Vehicle(
    std::string _name, const YAML::Node& cfg) :
        VehicleBase(_name), planner(KLevelPlanner::get_instance(cfg)) {
    YAML::Node vehicle_info = cfg["vehicle_list"][_name];
    level = vehicle_info["level"].as<int>();
    color = vehicle_info["color"].as<std::string>();
    init_x_min = vehicle_info["init"]["x"]["min"].as<double>();
    init_x_max = vehicle_info["init"]["x"]["max"].as<double>();
    init_y_min = vehicle_info["init"]["y"]["min"].as<double>();
    init_y_max = vehicle_info["init"]["y"]["max"].as<double>();
    init_v_min = vehicle_info["init"]["v"]["min"].as<double>();
    init_v_max = vehicle_info["init"]["v"]["max"].as<double>();
    init_yaw = vehicle_info["init"]["yaw"].as<double>();
    target.x = vehicle_info["target"]["x"].as<double>();
    target.y = vehicle_info["target"]["y"].as<double>();
    target.yaw = vehicle_info["target"]["yaw"].as<double>();
    vis_text_pos.x = vehicle_info["text"]["x"].as<double>();
    vis_text_pos.y = vehicle_info["text"]["y"].as<double>();

    vehicle_box2d = VehicleBase::get_box2d(state);
    safezone = VehicleBase::get_safezone(state);
    dt = cfg["delta_t"].as<double>();

    reset();
}

void Vehicle::reset(void) {
    footprint.clear();
    cur_action = Action::MAINTAIN;
    excepted_traj = StateList();
    have_got_target = false;

    state.x = Random::uniform(init_x_min, init_x_max);
    state.y = Random::uniform(init_y_min, init_y_max);
    state.v = Random::uniform(init_v_min, init_v_max);
    state.yaw = init_yaw;
    footprint.push_back(state);
}

void Vehicle::excute(std::vector<VehicleBase> others) {
    if (is_get_target()) {
        have_got_target = true;
        state.v = 0;
        cur_action = Action::MAINTAIN;
        excepted_traj = StateList();
    } else {
        std::pair<Action, StateList> act_and_traj = planner.planning(*this, others);
        cur_action = act_and_traj.first;
        excepted_traj = act_and_traj.second;
        state = utils::kinematic_propagate(state, utils::get_action_value(cur_action), dt);
        footprint.push_back(state);
    }
}

void Vehicle::draw_vehicle(bool fill_mode /* = false */) {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> head;
    Eigen::Matrix2d rot;
    head << 0.3 * VehicleBase::length, 0.3 * VehicleBase::length,
            VehicleBase::width/2, -VehicleBase::width/2;
    rot << cos(state.yaw), -sin(state.yaw), sin(state.yaw), cos(state.yaw);

    head = rot * head;
    head += Eigen::Vector2d(state.x, state.y).replicate(1, 2);
    vehicle_box2d = VehicleBase::get_box2d(state);

    std::vector<std::vector<double>> box2d_vec(2);
    std::vector<std::vector<double>> head_vec(2);
    box2d_vec[0].assign(vehicle_box2d.row(0).data(), vehicle_box2d.row(0).data() + vehicle_box2d.cols());
    box2d_vec[1].assign(vehicle_box2d.row(1).data(), vehicle_box2d.row(1).data() + vehicle_box2d.cols());
    head_vec[0].assign(head.row(0).data(), head.row(0).data() + head.cols());
    head_vec[1].assign(head.row(1).data(), head.row(1).data() + head.cols());

    if (!fill_mode) {
        plt::plot(box2d_vec[0], box2d_vec[1], color);
        plt::plot(head_vec[0], head_vec[1], color);
    } else {
        plt::fill(box2d_vec[0], box2d_vec[1], {{"color", color}, {"alpha", "0.5"}});
    }
}

void VehicleList::set_track_objects(void) {

}

void VehicleList::reset(void) {
    for (std::shared_ptr<Vehicle>& vehicle : vehicle_list) {
        vehicle->reset();
    }
}

bool VehicleList::is_all_get_target(void) {
    bool all_get_target = std::all_of(vehicle_list.begin(), vehicle_list.end(), 
                            [](const std::shared_ptr<Vehicle> vehicle) {return vehicle->is_get_target();});

    return all_get_target;
}

bool VehicleList::is_any_collision(void) {
    for (int i = 0; i < vehicle_list.size() - 1; ++i) {
        for (int j = i + 1; j < vehicle_list.size(); ++j) {
            if (utils::has_overlap(
                    VehicleBase::get_box2d(vehicle_list[i]->state),
                    VehicleBase::get_box2d(vehicle_list[j]->state))) {
                return true;
            }
        }
    }

    return false;
}

void VehicleList::push_back(std::shared_ptr<Vehicle> vehicle) {
    vehicle_list.push_back(vehicle);
}

void VehicleList::pop_back(void) {
    vehicle_list.pop_back();
}

std::vector<VehicleBase> VehicleList::exclude(int ego_idx) {
    std::shared_ptr<Vehicle> ego_vehicle = vehicle_list[ego_idx];
    return exclude(ego_vehicle);
}

std::vector<VehicleBase> VehicleList::exclude(std::shared_ptr<Vehicle> ego) {
    std::vector<VehicleBase> exclude_list;
    for (std::shared_ptr<Vehicle> vehicle : vehicle_list) {
        if (vehicle->name != ego->name) {
            exclude_list.push_back(*vehicle);
        }
    }
    
    return exclude_list;
}
