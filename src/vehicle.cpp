#include <vector>
#include <spdlog/spdlog.h>
#include <matplotlib-cpp/matplotlibcpp.h>
#include <fmt/core.h>
#include "vehicle.hpp"

namespace plt = matplotlibcpp;

std::pair<Action, StateList> Vehicle::excute(VehicleBase other) {
    std::pair<Action, StateList> act_and_traj;

    if (is_get_target()) {
        have_got_target = true;
        state.v = 0;
        act_and_traj.first = Action::MAINTAIN;
        act_and_traj.second = StateList();
    } else {
        act_and_traj = planner.planning(*this, other);
    }

    return act_and_traj;
}

void Vehicle::draw_vehicle(bool fill_mode /* = false */) {
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> head;
    Eigen::Matrix2d rot;
    head << 0.3 * VehicleBase::length, 0.3 * VehicleBase::length, VehicleBase::width/2, -VehicleBase::width/2;
    rot << cos(state.yaw), -sin(state.yaw), sin(state.yaw), cos(state.yaw);

    head = rot * head;
    head += Eigen::Vector2d(state.x, state.y).replicate(1, 2);
    vehicle_box2d = VehicleBase::get_box2d(state);

    std::vector<std::vector<double>> box2d_vec(2);
    std::vector<std::vector<double>> head_vec(2);
    box2d_vec[0].assign(vehicle_box2d.row(0).data(), vehicle_box2d.row(0).data() + vehicle_box2d.cols());
    box2d_vec[1].assign(vehicle_box2d.row(1).data(), vehicle_box2d.row(1).data() + vehicle_box2d.cols());
    head_vec[0].assign(head.row(0).data(), head.row(0).data() + head.cols());
    head_vec[1].assign(head.row(1).data(), head.row(1).data() + head.cols());

    if (!fill_mode) {
        plt::plot(box2d_vec[0], box2d_vec[1], color);
        plt::plot(head_vec[0], head_vec[1], color);
    } else {
        plt::fill(box2d_vec[0], box2d_vec[1], {{"color", color}, {"alpha", "0.8"}});
    }
}
