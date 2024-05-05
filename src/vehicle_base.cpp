#include <spdlog/spdlog.h>

#include "vehicle_base.hpp"

double VehicleBase::length = 5;
double VehicleBase::width = 2;
double VehicleBase::safe_length = 8;
double VehicleBase::safe_width = 2.4;
std::shared_ptr<EnvCrossroads> VehicleBase::env = nullptr;

void VehicleBase::set_target(State tar) {
    if (tar.x >= -25 && tar.x <= 25 && tar.y >= -25 && tar.y <= 25) {
        target = tar;
    } else {
        spdlog::error("set_target error, the target range must >= -25 and <= 25 !");
    }
}

void VehicleBase::set_level(int l) {
    if (l >= 0 && l < 3) {
        level = l;
    } else {
        spdlog::error("set_level error, the level must be >= 0 and > 3 !");
    }
}

bool VehicleBase::is_get_target(void) const {
    return have_got_target || hypot(state.x - target.x, state.y - target.y) < 1.7;
}
