/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-05-26 21:14:51
 * @LastEditTime: 2024-10-31 01:00:25
 * @FilePath: /vehicle-interaction-decision-making/include/planner.hpp
 * Copyright 2024 puyu, All Rights Reserved.
 */

#pragma once
#ifndef __PLANNER_HPP
#define __PLANNER_HPP

#include <yaml-cpp/yaml.h>

#include "utils.hpp"
#include "vehicle_base.hpp"

class MonteCarloTreeSearch {
private:
    /* data */
public:
    static double EXPLORATE_RATE;
    static double LAMDA;
    static double WEIGHT_AVOID;
    static double WEIGHT_SAFE;
    static double WEIGHT_OFFROAD;
    static double WEIGHT_DIRECTION;
    static double WEIGHT_DISTANCE;
    static double WEIGHT_VELOCITY;

    std::vector<StateList> other_predict_traj;
    uint64_t computation_budget;
    double dt;

    MonteCarloTreeSearch(const std::vector<StateList>& other_traj, const YAML::Node& cfg);
    ~MonteCarloTreeSearch() {}

    static void initialize(const YAML::Node& cfg) {
        MonteCarloTreeSearch::LAMDA = cfg["lamda"].as<double>();
        MonteCarloTreeSearch::WEIGHT_AVOID = cfg["weight_avoid"].as<double>();
        MonteCarloTreeSearch::WEIGHT_SAFE = cfg["weight_safe"].as<double>();
        MonteCarloTreeSearch::WEIGHT_OFFROAD = cfg["weight_offroad"].as<double>();
        MonteCarloTreeSearch::WEIGHT_DIRECTION = cfg["weight_direction"].as<double>();
        MonteCarloTreeSearch::WEIGHT_DISTANCE = cfg["weight_distance"].as<double>();
        MonteCarloTreeSearch::WEIGHT_VELOCITY = cfg["weight_velocity"].as<double>();
    }
    static bool is_opposite_direction(State pos, Eigen::MatrixXd ego_box2d);
    static double calc_cur_value(std::shared_ptr<Node> node, double last_node_value);

    std::shared_ptr<Node> excute(std::shared_ptr<Node> root);
    std::shared_ptr<Node> tree_policy(std::shared_ptr<Node> node);
    std::shared_ptr<Node> expand(std::shared_ptr<Node> node);
    std::shared_ptr<Node> get_best_child(std::shared_ptr<Node> node, double scalar);
    double default_policy(std::shared_ptr<Node> node);
    void update(std::shared_ptr<Node> node, double r);

};

class KLevelPlanner {
private:
    int steps;
    YAML::Node config;

    KLevelPlanner(const KLevelPlanner &single) = delete;
    const KLevelPlanner &operator=(const KLevelPlanner &single) = delete;
    KLevelPlanner(const YAML::Node& cfg) : config(cfg) {
        steps = cfg["max_step"].as<int>();
    }
    ~KLevelPlanner() {}
public:
    static KLevelPlanner& get_instance(const YAML::Node& cfg) {
        static KLevelPlanner planner_(cfg);
        return planner_;
    }

    std::pair<Action, StateList> planning(VehicleBase& ego) const;
    std::pair<std::vector<Action>, StateList> forward_simulate(
        const VehicleBase& ego, const std::vector<StateList>& traj) const;
    std::vector<StateList> get_prediction(
        const VehicleBase& ego, const std::vector<VehicleBase>& others) const;
};


#endif
