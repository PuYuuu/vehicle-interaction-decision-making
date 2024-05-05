#include <vector>
#include <spdlog/spdlog.h>
#include <matplotlib-cpp/matplotlibcpp.h>
#include <fmt/core.h>
#include "vehicle.hpp"

namespace plt = matplotlibcpp;

void Vehicle::set_level(int l) {
    if (l >= 0 && l < 3) {
        level = l;
    } else {
        spdlog::error("set_level error, the level must be >= 0 and > 3 !");
    }
}

void Vehicle::set_target(State tar) {
    if (tar.x >= -25 && tar.x <= 25 && tar.y >= -25 && tar.y <= 25) {
        target = tar;
    } else {
        spdlog::error("set_target error, the target range must >= -25 and <= 25 !");
    }
}

void Vehicle::excute(VehicleBase other) {
    if (is_get_target()) {
        have_got_target = true;
        state.v = 0;
    } else {

    }
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

bool Vehicle::is_get_target(void) {
    return have_got_target || hypot(state.x - target.x, state.y - target.y) < 1.7;
}
