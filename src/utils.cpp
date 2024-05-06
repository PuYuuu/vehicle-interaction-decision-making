#include <cmath>
#include <array>
#include <random>
#include <unordered_map>

#include "utils.hpp"

int Node::MAX_LEVEL = 6;
double (*Node::calc_value_callback)(std::shared_ptr<Node>, double) = nullptr;

constexpr std::array<const char*, 6> ACTIONNAMES = {
    "MAINTAIN",
    "TURNLEFT",
    "TURNRIGHT",
    "ACCELERATE",
    "DECELERATE",
    "BRAKE"
};

std::string get_action_name(Action action) {
    return ACTIONNAMES[static_cast<size_t>(action)];
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
    return children.size() >= ACTION_SIZE;
}

std::shared_ptr<Node> Node::add_child(Action next_action, double delta_t, StateList others, std::shared_ptr<Node> p) {
    State new_state = kinematic_propagate(state, get_action_value(next_action), delta_t);
    std::shared_ptr<Node> child = std::make_shared<Node>(
        new_state, cur_level + 1, p, next_action, others, goal_pose);
    child->actions = actions;
    child->actions.push_back(next_action);
    Node::calc_value_callback(child, value);
    children.push_back(child);

    return child;
}

std::shared_ptr<Node> Node::next_node(double delta_t, StateList others) {
    Action next_action = get_random_action();
    State new_state = kinematic_propagate(state, get_action_value(next_action), delta_t);
    std::shared_ptr<Node> node = std::make_shared<Node>(
        new_state, cur_level + 1, nullptr, next_action, others, goal_pose);
    Node::calc_value_callback(node, value);

    return node;
}

Eigen::Vector2d get_action_value(Action act) {
    static std::unordered_map<Action, Eigen::Vector2d> ACTION_MAP = {
        {Action::MAINTAIN, {0, 0}}, {Action::TURNLEFT, {0, M_PI_4}}, 
        {Action::TURNRIGHT, {0, -M_PI_4}}, {Action::ACCELERATE, {2.5, 0}}, 
        {Action::DECELERATE, {-2.5, 0}}, {Action::BRAKE, {-5, 0}}
    };

    return ACTION_MAP[act];
}

Action get_random_action(void) {
    std::mt19937 engine(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, ACTION_SIZE - 1);
    int random_number = dist(engine);

    return static_cast<Action>(random_number);
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
            double project = separating_axis[0] * box2d_1(0, j) + separating_axis[1] * box2d_1(1, j);
            box2d_min = std::min(box2d_min, project);
            box2d_max = std::max(box2d_max, project);
        }

        if (vehicle_min > box2d_max || box2d_min > vehicle_max) {
            return false;
        }
    }

    return true;
}

State kinematic_propagate(State state, Eigen::Vector2d act, double dt)
{
    State next_state;
    double acc = act[0];
    double omega = act[1];

    next_state.x = state.x + state.v * cos(state.yaw) * dt;
    next_state.y = state.y + state.v * sin(state.yaw) * dt;
    next_state.v = state.v + acc * dt;
    next_state.yaw = state.yaw + omega * dt;

    if (next_state.yaw > 2 * M_PI) {
        next_state.yaw -= 2 * M_PI;
    }
    if (next_state.yaw < 0) {
        next_state.yaw += 2 * M_PI;
    }

    if (next_state.v > 20) {
        next_state.v = 20;
    } else if (next_state.v < -20) {
        next_state.v = -20;
    }

    return next_state;
}
