/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-05-28 01:17:14
 * @LastEditTime: 2024-10-31 01:00:41
 * @FilePath: /vehicle-interaction-decision-making/include/vehicle.hpp
 * Copyright 2024 puyu, All Rights Reserved.
 */

#pragma once
#ifndef __VEHICLE_HPP
#define __VEHICLE_HPP

#include <set>
#include <string>

#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

#include "matplotlibcpp.h"
#include "utils.hpp"
#include "vehicle_base.hpp"
#include "planner.hpp"

class Vehicle : public VehicleBase {
private:
    double dt;
    KLevelPlanner& planner;
    double init_x_min;
    double init_x_max;
    double init_y_min;
    double init_y_max;
    double init_v_min;
    double init_v_max;
    double init_yaw;
    static int global_vehicle_idx;
    static PyObject* imshow_func;

    struct Outlook {
        int rows;
        int cols;
        int colors;
        std::vector<float> data;
    };
    Outlook outlook;
    void imshow(const Outlook& out, const State& state, std::vector<double> para);
public:
    std::string color;
    Action cur_action;
    StateList excepted_traj;
    std::vector<State> footprint;
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> vehicle_box2d;
    Eigen::Matrix<double, 2, 5, Eigen::RowMajor> safezone;
    State vis_text_pos;

    Vehicle(std::string _name, const YAML::Node& cfg);
    ~Vehicle() {}

    void reset(void);
    void excute(void);
    void draw_vehicle(std::string draw_style = "realistic", bool fill_mode = false);
    bool operator==(const Vehicle& other) const {
        return name == other.name;
    }
    bool operator!=(const Vehicle& other) const {
        return name != other.name;
    }
};

class VehicleList {
private:
    std::vector<std::shared_ptr<Vehicle>> vehicle_list;
    std::set<std::string> vehicle_names;
public:
    VehicleList() {
        vehicle_list.clear();
        vehicle_names.clear();
    }
    VehicleList(std::vector<std::shared_ptr<Vehicle>> vehicles) : vehicle_list(vehicles) { }
    ~VehicleList() {}

    size_t size(void) {
        return vehicle_list.size();
    }
    bool is_all_get_target(void);
    bool is_any_collision(void);
    void push_back(std::shared_ptr<Vehicle> vehicle);
    void pop_back(void);
    void reset(void);
    void set_track_objects(void);
    void update_track_objects(void);
    std::vector<VehicleBase> exclude(int ego_idx);
    std::vector<VehicleBase> exclude(std::shared_ptr<Vehicle> ego);
    std::shared_ptr<Vehicle> operator[](size_t index) {
        return vehicle_list[index];
    }
    std::shared_ptr<Vehicle> operator[](std::string name);
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
