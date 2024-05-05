#include <cmath>
#include <unordered_map>

#include "utils.hpp"

enum class Action {MAINTAIN, TURNLEFT, TURNRIGHT, ACCELERATE, DECELERATE, BRAKE};

Eigen::Vector2d get_action_value(Action act) {
    static std::unordered_map<Action, Eigen::Vector2d> ACTION_MAP = {
        {Action::MAINTAIN, {0, 0}}, {Action::TURNLEFT, {0, M_PI_4}}, 
        {Action::TURNRIGHT, {0, -M_PI_4}}, {Action::ACCELERATE, {2.5, 0}}, 
        {Action::DECELERATE, {-2.5, 0}}, {Action::BRAKE, {-5, 0}}
    };

    return ACTION_MAP[act];
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
