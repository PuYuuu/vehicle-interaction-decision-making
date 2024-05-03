#pragma once
#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <vector>

class State {
private:
    /* data */
public:
    double x;
    double y;
    double yaw;
    double v;

    State() {}
    State(double _x, double _y, double _yaw, double _v) :
        x(_x), y(_y), yaw(_yaw), v(_v) {}
    ~State() {}

    std::vector<double> to_vector(void) {
        return std::vector<double>{x, y, yaw, v};
    }
};

#endif
