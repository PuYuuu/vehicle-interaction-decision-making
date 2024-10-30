/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-05-26 21:14:51
 * @LastEditTime: 2024-10-31 00:59:50
 * @FilePath: /vehicle-interaction-decision-making/src/planner.cpp
 * Copyright 2024 puyu, All Rights Reserved.
 */

#include <unordered_set>
#include <spdlog/spdlog.h>
#include <fmt/core.h>

#include "planner.hpp"

constexpr double TWO_PI = M_PI * 2;

double MonteCarloTreeSearch::EXPLORATE_RATE = 1 / (2 * sqrt(2.0));
double MonteCarloTreeSearch::LAMDA = 0.9;
double MonteCarloTreeSearch::WEIGHT_AVOID = 10;
double MonteCarloTreeSearch::WEIGHT_SAFE = 0.2;
double MonteCarloTreeSearch::WEIGHT_OFFROAD = 2;
double MonteCarloTreeSearch::WEIGHT_DIRECTION = 1;
double MonteCarloTreeSearch::WEIGHT_DISTANCE = 0.1;
double MonteCarloTreeSearch::WEIGHT_VELOCITY = 0.05;

MonteCarloTreeSearch::MonteCarloTreeSearch(const std::vector<StateList>& other_traj, const YAML::Node& cfg) {
    computation_budget = cfg["computation_budget"].as<uint64_t>();
    dt = cfg["delta_t"].as<double>();

    for (int i = 0; i < Node::MAX_LEVEL + 1; ++i) {
        StateList other_states;
        for (const StateList states : other_traj) {
            other_states.push_back(states[i]);
        }
        other_predict_traj.emplace_back(other_states);
    }
}

double MonteCarloTreeSearch::calc_cur_value(std::shared_ptr<Node> node, double last_node_value) {
    double x = node->state.x;
    double y = node->state.y;
    double yaw = node->state.yaw;
    double velocity = node->state.v;
    int step = node->cur_level;
    Eigen::Matrix<double, 2, 5> ego_box2d = VehicleBase::get_box2d(node->state);
    Eigen::Matrix<double, 2, 5> ego_safezone = VehicleBase::get_safezone(node->state);

    int avoid = 0;
    int safe = 0;
    for (auto& cur_other_state : node->other_agent_state) {
        if (utils::has_overlap(ego_box2d, VehicleBase::get_box2d(cur_other_state))) {
            avoid = -1;
        }
        if (utils::has_overlap(ego_safezone, VehicleBase::get_safezone(cur_other_state))) {
            safe = -1;
        }
    }

    int offroad = 0;
    for (auto& rect : VehicleBase::env->rect_mat) {
        if (utils::has_overlap(ego_box2d, rect)) {
            offroad = -1;
            break;
        }
    }

    int direction = 0;
    if (MonteCarloTreeSearch::is_opposite_direction(node->state, ego_box2d)) {
        direction = -1;
    }
    
    double delta_yaw = std::fmod(abs(yaw - node->goal_pose.yaw), TWO_PI);
    delta_yaw = std::min(delta_yaw, TWO_PI - delta_yaw);
    double distance = -(abs(x - node->goal_pose.x) + abs(y - node->goal_pose.y) +
                        1.5 * delta_yaw);

    double cur_reward = MonteCarloTreeSearch::WEIGHT_AVOID * avoid +
                        MonteCarloTreeSearch::WEIGHT_SAFE * safe +
                        MonteCarloTreeSearch::WEIGHT_OFFROAD * offroad +
                        MonteCarloTreeSearch::WEIGHT_DISTANCE * distance +
                        MonteCarloTreeSearch::WEIGHT_DIRECTION * direction +
                        MonteCarloTreeSearch::WEIGHT_VELOCITY * velocity;
    double total_reward = last_node_value + pow(MonteCarloTreeSearch::LAMDA, (step - 1)) * cur_reward;
    node->value = total_reward;

    return total_reward;
}

bool MonteCarloTreeSearch::is_opposite_direction(State pos, Eigen::MatrixXd ego_box2d) {
    double x = pos.x;
    double y = pos.y;
    double yaw = pos.yaw;

    for (auto laneline : VehicleBase::env->laneline_mat) {
        if (utils::has_overlap(ego_box2d, laneline)) {
            return true;
        }
    }

    double lanewidth = VehicleBase::env->lanewidth;
    if (x > -lanewidth && x < 0 && (y < -lanewidth || y > lanewidth)) {
        // down lane
        if (yaw > 0 && yaw < M_PI) {
            return true;
        }
    } else if (x > 0 && x < lanewidth && (y < -lanewidth || y > lanewidth)) {
        // up lane
        if (!(yaw > 0 && yaw < M_PI)) {
            return true;
        }
    } else if (y > -lanewidth && y < 0 && (x < -lanewidth || x > lanewidth)) {
        // right lane
        if (yaw > M_PI_2 && yaw < 3 * M_PI_2) {
            return true;
        }
    } else if (y > 0 && y < lanewidth && (x < -lanewidth || x > lanewidth)) {
        // left lane
        if (!(yaw > M_PI_2 && yaw < 3 * M_PI_2)) {
            return true;
        }
    }

    return false;
}

std::shared_ptr<Node> MonteCarloTreeSearch::excute(std::shared_ptr<Node> root) {
    for (uint64_t iter = 0; iter < computation_budget; ++iter) {
        // 1. Find the best node to expand
        std::shared_ptr<Node> expand_node = tree_policy(root);
        // 2. Random run to add node and get reward
        double reward = default_policy(expand_node);
        // 3. Update all passing nodes with reward
        update(expand_node, reward);
    }

    return get_best_child(root, 0);
}

std::shared_ptr<Node> MonteCarloTreeSearch::tree_policy(std::shared_ptr<Node> node) {
    while (node->is_terminal() == false) {
        if (node->children.empty()) {
            return expand(node);
        } else if (Random::uniform(0.0, 1.0) < 0.5) {
            node = get_best_child(node, MonteCarloTreeSearch::EXPLORATE_RATE);
        } else {
            if (node->is_fully_expanded() == false) {
                return expand(node);
            } else {
                node = get_best_child(node, MonteCarloTreeSearch::EXPLORATE_RATE);
            }
        }
    }

    return node;
}

std::shared_ptr<Node> MonteCarloTreeSearch::expand(std::shared_ptr<Node> node) {
    std::unordered_set<Action> tried_actions;
    for (auto child : node->children) {
        tried_actions.insert(child->action);
    }

    Action next_action = Random::choice(ACTION_LIST);
    while (!node->is_terminal() && tried_actions.count(next_action)) {
        next_action = Random::choice(ACTION_LIST);
    }
    StateList other_states = other_predict_traj[node->cur_level + 1];
    node->add_child(next_action, dt, other_states);

    return node->children.back();
}

std::shared_ptr<Node> MonteCarloTreeSearch::get_best_child(std::shared_ptr<Node> node, double scalar) {
    double best_score = -INFINITY;
    std::vector<std::shared_ptr<Node>> best_children;

    for (auto child : node->children) {
        double exploit = child->reward / child->visits;
        double explore = sqrt(2 * log(node->visits) / child->visits);
        double score = exploit + scalar + explore;
        if (score == best_score) {
            best_children.push_back(child);
        } else if (score > best_score) {
            best_children = {child};
            best_score = score;
        }
    }
    if (best_children.empty()) {
        return node;
    }

    return Random::choice(best_children);
}

double MonteCarloTreeSearch::default_policy(std::shared_ptr<Node> node) {
    while (!node->is_terminal()) {
        StateList other_states = other_predict_traj[node->cur_level + 1];
        std::shared_ptr<Node> next_node = node->next_node(dt, other_states);
        node = next_node;
    }

    return node->value;
}

void MonteCarloTreeSearch::update(std::shared_ptr<Node> node, double r) {
    while (node != nullptr) {
        node->visits += 1;
        node->reward += r;
        node = node->parent;
    }
}

std::pair<Action, StateList> KLevelPlanner::planning(VehicleBase& ego) const {
    std::vector<VehicleBase> others;
    for (const TrackedObject& obj : ego.tracked_objects) {
        VehicleBase other(obj.name);
        other.state = obj.state;
        other.target = obj.target;
        other.have_got_target = other.is_get_target();
        others.emplace_back(other);
    }

    std::vector<StateList> other_prediction = get_prediction(ego, others);

    for (size_t i = 0; i < others.size(); ++i) {
        PredictTraj predict_traj{1.0, other_prediction[i]};
        ego.tracked_objects[i].predict_trajs.clear();
        ego.tracked_objects[i].predict_trajs.emplace_back(predict_traj);
    }

    std::pair<std::vector<Action>, StateList> ret = forward_simulate(ego, other_prediction);

    return std::make_pair(ret.first[0], ret.second);
}

std::pair<std::vector<Action>, StateList> KLevelPlanner::forward_simulate(
                                const VehicleBase& ego, const std::vector<StateList>& traj) const {
    MonteCarloTreeSearch mcts(traj, config);
    std::shared_ptr<Node> current_node =
        std::make_shared<Node>(ego.state, 0, nullptr, Action::MAINTAIN, StateList(), ego.target);
    current_node = mcts.excute(current_node);
    for (int i = 0; i < Node::MAX_LEVEL - 1; ++i) {
        current_node = mcts.get_best_child(current_node, 0);
    }

    std::vector<Action> actions = current_node->actions;
    StateList expected_traj;
    while (current_node != nullptr) {
        expected_traj.push_back(current_node->state);
        current_node = current_node->parent;
    }
    expected_traj.reverse();

    if (expected_traj.size() < steps + 1) {
        spdlog::debug(
            "The max level of the node is not enough({}),using the last value to complete it.",
            expected_traj.size());
        expected_traj.expand(steps + 1);
    }

    return std::make_pair(actions, expected_traj);
}

std::vector<StateList> KLevelPlanner::get_prediction(
    const VehicleBase& ego, const std::vector<VehicleBase>& others) const {
    std::vector<StateList> pred_trajectory;

    if (ego.level == 0) {
        for (const VehicleBase& other : others) {
            StateList pred_traj;
            for (size_t i = 0; i < steps + 1; ++i) {
                pred_traj.push_back(other.state);
            }
            pred_trajectory.emplace_back(pred_traj);
        }
    } else if (ego.level > 0) {
        for (size_t idx = 0; idx < others.size(); ++idx) {
            if (others[idx].have_got_target) {
                StateList pred_traj;
                for (size_t i = 0; i < steps + 1; ++i) {
                    pred_traj.push_back(others[idx].state);
                }
                pred_trajectory.emplace_back(pred_traj);
                continue;
            }
            VehicleBase exchanged_ego = others[idx];
            exchanged_ego.level = ego.level - 1;
            std::vector<VehicleBase> exchanged_others = {ego};
            for (size_t i = 0; i < others.size(); ++i) {
                if (i != idx) {
                    exchanged_others.push_back(others[i]);
                }
            }
            std::vector<StateList> exchage_pred_others = get_prediction(exchanged_ego, exchanged_others);
            auto pred_idx_vechicle = forward_simulate(exchanged_ego, exchage_pred_others);
            pred_trajectory.emplace_back(pred_idx_vechicle.second);
        }
    } else {
        spdlog::error("get_prediction() excute error, the level must be >= 0 and > 3 !");
    }

    return pred_trajectory;
}
