/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-05-26 21:14:51
 * @LastEditTime: 2024-10-31 00:59:56
 * @FilePath: /vehicle-interaction-decision-making/src/utils.cpp
 * Copyright 2024 puyu, All Rights Reserved.
 */

#include <cmath>
#include <array>
#include <fstream>
#include <unordered_map>

#include <spdlog/spdlog.h>

#include "utils.hpp"

int Node::MAX_LEVEL = 6;
// double (*Node::calc_value_callback)(std::shared_ptr<Node>, double) = nullptr;
std::function<double(std::shared_ptr<Node>, double)> Node::calc_value_callback;

std::default_random_engine Random::engine(std::random_device{}());

constexpr std::array<const char*, 6> ACTIONNAMES = {
    "MAINTAIN",
    "TURNLEFT",
    "TURNRIGHT",
    "ACCELERATE",
    "DECELERATE",
    "BRAKE"
};

int Random::uniform(int _min, int _max) {
    std::uniform_int_distribution dist(_min, _max);
    return dist(Random::engine);
}

double Random::uniform(double _min, double _max) {
    std::uniform_real_distribution<double> dist(_min, _max);
    return dist(Random::engine);
}

Node::Node(State _state, int _level, std::shared_ptr<Node> p,
            Action act, StateList others, State goal) :
            state(_state), cur_level(_level), parent(p), action(act),
            other_agent_state(others), goal_pose(goal) {
    value = 0.0;
    reward = 0.0;
    visits = 0;
}

bool Node::is_terminal(void) {
    return cur_level >= Node::MAX_LEVEL;
}

bool Node::is_fully_expanded(void) {
    return children.size() >= ACTION_LIST.size();
}

std::shared_ptr<Node> Node::add_child(Action next_action, double delta_t, StateList others) {
    State new_state = utils::kinematic_propagate(state, utils::get_action_value(next_action), delta_t);
    std::shared_ptr<Node> child = std::make_shared<Node>(
        new_state, cur_level + 1, shared_from_this(), next_action, others, goal_pose);
    child->actions = actions;
    child->actions.push_back(next_action);
    if (Node::calc_value_callback) {
        Node::calc_value_callback(child, value);
    } else {
        spdlog::error("Node::calc_value_callback is null, which will lead to a program exception !");
        std::exit(EXIT_FAILURE);
    }
    children.push_back(child);

    return child;
}

std::shared_ptr<Node> Node::next_node(double delta_t, StateList others) {
    Action next_action = Random::choice(ACTION_LIST);
    State new_state = utils::kinematic_propagate(state, utils::get_action_value(next_action), delta_t);
    std::shared_ptr<Node> node = std::make_shared<Node>(
        new_state, cur_level + 1, nullptr, next_action, others, goal_pose);
    if (Node::calc_value_callback) {
        Node::calc_value_callback(node, value);
    } else {
        spdlog::error("Node::calc_value_callback is null !");
    }

    return node;
}

namespace utils {

    std::string get_action_name(Action action) {
        return ACTIONNAMES[static_cast<size_t>(action)];
    }

    Eigen::Vector2d get_action_value(Action act) {
        static std::unordered_map<Action, Eigen::Vector2d> ACTION_MAP = {
            {Action::MAINTAIN, {0, 0}}, {Action::TURNLEFT, {0, M_PI_4}}, 
            {Action::TURNRIGHT, {0, -M_PI_4}}, {Action::ACCELERATE, {2.5, 0}}, 
            {Action::DECELERATE, {-2.5, 0}}, {Action::BRAKE, {-5, 0}}
        };

        return ACTION_MAP[act];
    }

    bool has_overlap(Eigen::MatrixXd box2d_0, Eigen::MatrixXd box2d_1) {
        std::vector<Eigen::Vector2d> total_sides;
        for (size_t idx = 1; idx < box2d_0.cols(); ++idx) {
            Eigen::Vector2d tmp = box2d_0.col(idx).head(2) - box2d_0.col(idx - 1).head(2);
            total_sides.emplace_back(tmp);
        }
        for (size_t idx = 1; idx < box2d_1.cols(); ++idx) {
            Eigen::Vector2d tmp = box2d_1.col(idx).head(2) - box2d_1.col(idx - 1).head(2);
            total_sides.emplace_back(tmp);
        }

        for (size_t idx = 0; idx < total_sides.size(); ++idx) {
            Eigen::Vector2d separating_axis;
            separating_axis[0] = -total_sides[idx][1];
            separating_axis[1] = total_sides[idx][0];
            
            double vehicle_min = INFINITY;
            double vehicle_max = -INFINITY;
            for (size_t j = 0; j < box2d_0.cols(); ++j) {
                double project = separating_axis[0] * box2d_0(0, j) + separating_axis[1] * box2d_0(1, j);
                vehicle_min = std::min(vehicle_min, project);
                vehicle_max = std::max(vehicle_max, project);
            }

            double box2d_min = INFINITY;
            double box2d_max = -INFINITY;
            for (size_t j = 0; j < box2d_1.cols(); ++j) {
                double project =
                    separating_axis[0] * box2d_1(0, j) + separating_axis[1] * box2d_1(1, j);
                box2d_min = std::min(box2d_min, project);
                box2d_max = std::max(box2d_max, project);
            }

            if (vehicle_min > box2d_max || box2d_min > vehicle_max) {
                return false;
            }
        }

        return true;
    }

    State kinematic_propagate(const State& state, Eigen::Vector2d act, double dt) {
        State next_state;
        double acc = act[0];
        double omega = act[1];

        next_state.x = state.x + state.v * cos(state.yaw) * dt;
        next_state.y = state.y + state.v * sin(state.yaw) * dt;
        next_state.v = state.v + acc * dt;
        next_state.yaw = state.yaw + omega * dt;

        while (next_state.yaw > 2 * M_PI) {
            next_state.yaw -= 2 * M_PI;
        }
        while (next_state.yaw < 0) {
            next_state.yaw += 2 * M_PI;
        }

        if (next_state.v > 20) {
            next_state.v = 20;
        } else if (next_state.v < -20) {
            next_state.v = -20;
        }

        return next_state;
    }

    std::string absolute_path(std::string path) {
        std::string abs_path = "";

        if (path.empty()) {
            return abs_path;
        }

        if (path[0] == '~') {
            const char* home_dir = getenv("HOME");
            if (!home_dir) {
                spdlog::error("environment variable HOME is not set!");
                return abs_path;
            }
            abs_path = home_dir + path.substr(1);
        } else if (path[0] == '.') {
            char tmp_buffer[40960] = {0};
            char* ret = realpath(path.c_str(), tmp_buffer);
            abs_path = std::string(tmp_buffer);
        } else {
            return path;
        }

        return abs_path;
    }

    std::vector<float> imread(std::string filename, int& rows, int& cols, int& colors) {
        std::vector<float> image;
        std::ifstream file(filename);

        if (!file.is_open()) {
            spdlog::error("open {} failed !", filename);
            return image;
        }

        std::string line;
        getline(file, line);
        if (line != "Convert from PNG") {
            spdlog::error("this format is not supported: {}", filename);
            return image;
        }
        getline(file, line);
        std::istringstream iss(line);
        iss >> rows >> cols >> colors;
        image.resize(rows * cols * colors);
        int idx = 0;
        while (getline(file, line)) {
            std::istringstream iss(line);
            for (int i = 0; i < colors; ++i) {
                iss >> image[idx++];
            }
        }
        file.close();

        // directly return will trigger RVO (Return Value Optimization)
        return std::move(image);
    }

}
