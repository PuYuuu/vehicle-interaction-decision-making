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
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle_box2d;
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> safezone;

    Vehicle(std::string _name, State _st, std::string _c, YAML::Node cfg) :
        VehicleBase(_name, _st, _c) {
        vehicle_box2d = VehicleBase::get_box2d(state);
        safezone = VehicleBase::get_safezone(state);
        dt = cfg["delta_t"].as<double>();
        planner = KLevelPlanner(cfg);
    }
    ~Vehicle() {}

    std::pair<Action, StateList> excute(VehicleBase other);
    void draw_vehicle(bool fill_mode = false);
};


#endif
