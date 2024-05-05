#include "vehicle_base.hpp"

double VehicleBase::length = 5;
double VehicleBase::width = 2;
double VehicleBase::safe_length = 8;
double VehicleBase::safe_width = 2.4;
std::shared_ptr<EnvCrossroads> VehicleBase::env = nullptr;
