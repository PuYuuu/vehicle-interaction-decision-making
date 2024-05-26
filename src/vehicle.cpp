#include <vector>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <matplotlib-cpp/matplotlibcpp.h>
#include <fmt/core.h>

#include "vehicle.hpp"

namespace plt = matplotlibcpp;

std::filesystem::path source_file_path(__FILE__);
std::filesystem::path vehicle_img_path =
    source_file_path.parent_path().parent_path() / "img" / "vehicle";
const std::vector<std::pair<std::string, std::string>> vehicle_show_config = {
    // {"#30A9DE", vehicle_img_path / "blue.png"},
    {"#0000FF", vehicle_img_path / "blue.png"},
    {"#E53A40", vehicle_img_path / "red.png"},
    {"#4CAF50", vehicle_img_path / "green.png"},
    {"#FFFF00", vehicle_img_path / "yellow.png"},
    {"#00FFFF", vehicle_img_path / "cyan.png"},
    {"#FA58F4", vehicle_img_path / "purple.png"},
    {"#000000", vehicle_img_path / "black.png"},
};

int Vehicle::global_vehicle_idx = 0;

Vehicle::Vehicle(
    std::string _name, const YAML::Node& cfg) :
        VehicleBase(_name), planner(KLevelPlanner::get_instance(cfg)) {
    YAML::Node vehicle_info = cfg["vehicle_list"][_name];
    level = vehicle_info["level"].as<int>();
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

    int local_loop_idx = Vehicle::global_vehicle_idx % vehicle_show_config.size();
    color = vehicle_show_config[local_loop_idx].first;
    std::string vehicle_pic_path = vehicle_show_config[local_loop_idx].second;
    // todo
    // outlook = plt.imread(vehicle_pic_path, format = "png");
    ++Vehicle::global_vehicle_idx;

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

void Vehicle::excute(void) {
    if (is_get_target()) {
        have_got_target = true;
        state.v = 0;
        cur_action = Action::MAINTAIN;
        excepted_traj = StateList();
    } else {
        std::pair<Action, StateList> act_and_traj = planner.planning(*this);
        cur_action = act_and_traj.first;
        excepted_traj = act_and_traj.second;
        state = utils::kinematic_propagate(state, utils::get_action_value(cur_action), dt);
        footprint.push_back(state);
    }
}

void Vehicle::draw_vehicle(std::string draw_style/* = "realistic"*/, bool fill_mode /* = false */) {
    draw_style = "linestyle";
    if (draw_style == "realistic") {
        // todo
    } else {
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
        box2d_vec[0].assign(vehicle_box2d.row(0).data(),
                            vehicle_box2d.row(0).data() + vehicle_box2d.cols());
        box2d_vec[1].assign(vehicle_box2d.row(1).data(),
                            vehicle_box2d.row(1).data() + vehicle_box2d.cols());
        head_vec[0].assign(head.row(0).data(), head.row(0).data() + head.cols());
        head_vec[1].assign(head.row(1).data(), head.row(1).data() + head.cols());

        if (!fill_mode) {
            plt::plot(box2d_vec[0], box2d_vec[1], color);
            plt::plot(head_vec[0], head_vec[1], color);
        } else {
            plt::fill(box2d_vec[0], box2d_vec[1], {{"color", color}, {"alpha", "0.5"}});
        }
    }
}

void VehicleList::reset(void) {
    for (std::shared_ptr<Vehicle>& vehicle : vehicle_list) {
        vehicle->reset();
    }
    update_track_objects();
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
    if (vehicle_names.count(vehicle->name) > 0) {
        spdlog::error(fmt::format("vehicle name [{}] duplication is not acceptable !", vehicle->name));
        std::exit(EXIT_FAILURE);
    } else {
        vehicle_list.push_back(vehicle);
        vehicle_names.insert(vehicle->name);
    }
}

void VehicleList::pop_back(void) {
    vehicle_names.erase(vehicle_list.back()->name);
    vehicle_list.pop_back();
}

std::shared_ptr<Vehicle> VehicleList::operator[](std::string name) {
    for (std::shared_ptr<Vehicle> vehicle : vehicle_list) {
        if (vehicle->name == name) {
            return vehicle;
        }
    }

    return nullptr;
}

void VehicleList::set_track_objects(void) {
    for (size_t i = 0; i < vehicle_list.size(); ++i) {
        for (size_t j = 0; j < vehicle_list.size(); ++j) {
            if (j != i) {
                TrackedObject track_object(vehicle_list[j]->name);
                track_object.state = vehicle_list[j]->state;
                track_object.target = vehicle_list[j]->target;
                vehicle_list[i]->tracked_objects.emplace_back(track_object);
            }
        }
    }
}

void VehicleList::update_track_objects(void) {
    for (std::shared_ptr<Vehicle>& vehicle : vehicle_list) {
        for (TrackedObject& object : vehicle->tracked_objects) {
            object.state = (*this)[object.name]->state;
        }
    }
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
