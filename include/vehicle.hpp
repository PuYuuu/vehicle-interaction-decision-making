#pragma once
#ifndef __VEHICLE_HPP
#define __VEHICLE_HPP

#include <string>

#include <yaml-cpp/yaml.h>
#include <Eigen/Core>

#include "utils.hpp"
#include "vehicle_base.hpp"
#include "planner.hpp"

class Vehicle : public VehicleBase
{
private:
    double dt;
    KLevelPlanner planner;
public:
    Action cur_action;
    StateList excepted_traj;
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle_box2d;
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> safezone;

    Vehicle(std::string _name, State _st, std::string _c, const YAML::Node& cfg) :
        VehicleBase(_name, _st, _c) {
        vehicle_box2d = VehicleBase::get_box2d(state);
        safezone = VehicleBase::get_safezone(state);
        dt = cfg["delta_t"].as<double>();
        planner = KLevelPlanner(cfg);
        cur_action = Action::MAINTAIN;
        excepted_traj = StateList();
    }
    ~Vehicle() {}

    std::pair<Action, StateList> excute(VehicleBase other);
    void draw_vehicle(bool fill_mode = false);
    bool operator==(const Vehicle& other) const {
        return name == other.name;
    }
    bool operator!=(const Vehicle& other) const {
        return name != other.name;
    }
};

class VehicleList
{
private:
    std::vector<std::shared_ptr<Vehicle>> vehicle_list;
public:
    VehicleList(std::vector<std::shared_ptr<Vehicle>> vehicles) : vehicle_list(vehicles) { }
    ~VehicleList() {}

    size_t size(void) {
        return vehicle_list.size();
    }
    bool is_all_get_target(void);
    bool is_any_collision(void);
    void push_back(std::shared_ptr<Vehicle> vehicle);
    void pop_back(void);
    std::vector<std::shared_ptr<Vehicle>> exclude(int ego_idx);
    std::vector<std::shared_ptr<Vehicle>> exclude(std::shared_ptr<Vehicle> ego);
    std::shared_ptr<Vehicle> operator[](size_t index) {
        return vehicle_list[index];
    }
    auto begin() {
        return vehicle_list.begin();
    }

    auto end() {
        return vehicle_list.end();
    }

    auto begin() const {
        return vehicle_list.begin();
    }

    auto end() const {
        return vehicle_list.end();
    }
};

#endif
