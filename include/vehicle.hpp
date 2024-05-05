#pragma once
#ifndef __VEHICLE_HPP
#define __VEHICLE_HPP

#include <string>

#include <yaml-cpp/yaml.h>
#include <Eigen/Core>

#include "utils.hpp"
#include "vehicle_base.hpp"

class Vehicle : public VehicleBase
{
private:
    bool have_got_target;
public:
    int level;
    State target;
    double dt;
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle_box2d;
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> safezone;

    Vehicle(std::string _name, State _st, std::string _c, YAML::Node cfg) :
        VehicleBase(_name, _st, _c), level(0) {
        vehicle_box2d = VehicleBase::get_box2d(state);
        safezone = VehicleBase::get_safezone(state);
        target = State(0, 0, 0, 0);
        have_got_target = false;
        dt = cfg["delta_t"].as<double>();
    }
    ~Vehicle() {}

    void set_level(int l);
    void set_target(State tar);
    void excute(VehicleBase other);
    void draw_vehicle(bool fill_mode = false);
    bool is_get_target(void);
};


#endif
