#pragma once
#ifndef __VEHICLE_BASE_HPP
#define __VEHICLE_BASE_HPP

#include <cmath>
#include <string>
#include <memory>

#include <Eigen/Core>

#include "env.hpp"
#include "utils.hpp"

class VehicleBase {
private:
    /* data */
public:
    static double length;    
    static double width;
    static double safe_length;
    static double safe_width;
    static std::shared_ptr<EnvCrossroads> env;

    std::string name;
    State state;
    std::string color;

    VehicleBase(std::string _name, State _st, std::string _c) :
        name(_name), state(_st), color(_c) {}
    ~VehicleBase() {};

    static void initialize(std::shared_ptr<EnvCrossroads> _env,
        double _len, double _width, double _safe_len, double _safe_width);
    static Eigen::Matrix<double, 2, 5> get_box2d(const State& tar_offset);
};

double VehicleBase::length = 5;
double VehicleBase::width = 2;
double VehicleBase::safe_length = 8;
double VehicleBase::safe_width = 2.4;
std::shared_ptr<EnvCrossroads> VehicleBase::env = nullptr;

void VehicleBase::initialize(std::shared_ptr<EnvCrossroads> _env,
        double _len, double _width, double _safe_len, double _safe_width) {
    VehicleBase::length = _len;
    VehicleBase::width = _width;
    VehicleBase::safe_length = _safe_len;
    VehicleBase::safe_width = _safe_width;
    VehicleBase::env = _env;
}

Eigen::Matrix<double, 2, 5> get_box2d(const State& tar_offset) {
    Eigen::Matrix<double, 2, 5> vehicle;
    vehicle << -VehicleBase::length/2, VehicleBase::length/2, VehicleBase::length/2, -VehicleBase::length/2, -VehicleBase::length/2,
             VehicleBase::width/2, VehicleBase::width/2, -VehicleBase::width/2, -VehicleBase::width/2, VehicleBase::width/2;
    Eigen::Matrix2d rot;
    rot << cos(tar_offset.yaw), -sin(tar_offset.yaw), sin(tar_offset.yaw), cos(tar_offset.yaw);
    
    vehicle = rot * vehicle;
    vehicle += Eigen::Vector2d(tar_offset.x, tar_offset.y).replicate(1, 5);

    return vehicle;
}

#endif
