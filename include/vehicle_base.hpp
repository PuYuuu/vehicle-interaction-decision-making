/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-05-17 23:21:15
 * @LastEditTime: 2024-10-31 01:00:37
 * @FilePath: /vehicle-interaction-decision-making/include/vehicle_base.hpp
 * Copyright 2024 puyu, All Rights Reserved.
 */

#pragma once
#ifndef __VEHICLE_BASE_HPP
#define __VEHICLE_BASE_HPP

#include <cmath>
#include <string>
#include <memory>

#include <Eigen/Core>

#include "env.hpp"
#include "utils.hpp"
#include "tracked_object.hpp"

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
    State target;
    int level;
    bool have_got_target;
    std::vector<TrackedObject> tracked_objects;

    VehicleBase(std::string _name) :
        name(_name), level(0), have_got_target(false) {
        state = State(0, 0, 0, 0);
        target = State(0, 0, 0, 0);
    }
    virtual ~VehicleBase() {};
    void set_target(State tar);
    void set_level(int l);
    bool is_get_target(void) const;

    static void initialize(std::shared_ptr<EnvCrossroads> _env,
        double _len, double _width, double _safe_len, double _safe_width) {
        VehicleBase::length = _len;
        VehicleBase::width = _width;
        VehicleBase::safe_length = _safe_len;
        VehicleBase::safe_width = _safe_width;
        VehicleBase::env = _env;
    }

    static Eigen::Matrix<double, 2, 5> get_box2d(const State& tar_offset) {
        Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle;
        vehicle << -VehicleBase::length/2, VehicleBase::length/2,
                    VehicleBase::length/2, -VehicleBase::length/2, -VehicleBase::length/2,
                    VehicleBase::width/2, VehicleBase::width/2,
                    -VehicleBase::width/2, -VehicleBase::width/2, VehicleBase::width/2;
        Eigen::Matrix2d rot;
        rot << cos(tar_offset.yaw), -sin(tar_offset.yaw), sin(tar_offset.yaw), cos(tar_offset.yaw);
        
        vehicle = rot * vehicle;
        vehicle += Eigen::Vector2d(tar_offset.x, tar_offset.y).replicate(1, 5);

        return vehicle;
    }

    static Eigen::Matrix<double, 2, 5> get_safezone(const State& tar_offset) {
        Eigen::Matrix<double, 2, 5, Eigen::RowMajor> safezone;
        safezone << -VehicleBase::safe_length/2, VehicleBase::safe_length/2,
                    VehicleBase::safe_length/2, -VehicleBase::safe_length/2, -VehicleBase::safe_length/2,
                    VehicleBase::safe_width/2, VehicleBase::safe_width/2,
                    -VehicleBase::safe_width/2, -VehicleBase::safe_width/2, VehicleBase::safe_width/2;
        Eigen::Matrix2d rot;
        rot << cos(tar_offset.yaw), -sin(tar_offset.yaw), sin(tar_offset.yaw), cos(tar_offset.yaw);

        safezone = rot * safezone;
        safezone += Eigen::Vector2d(tar_offset.x, tar_offset.y).replicate(1, 5);

        return safezone;
    }
};

#endif
