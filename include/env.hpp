/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-01-11 00:48:20
 * @LastEditTime: 2024-10-31 01:00:20
 * @FilePath: /vehicle-interaction-decision-making/include/env.hpp
 * Copyright 2024 puyu, All Rights Reserved.
 */

#pragma once
#ifndef __ENV_HPP
#define __ENV_HPP

#include <vector>
#include <Eigen/Core>

class EnvCrossroads {
private:

public:
    double map_size;
    double lanewidth;

    std::vector<std::vector<std::vector<double>>> rect;
    std::vector<std::vector<std::vector<double>>> laneline;

    std::vector<Eigen::MatrixXd> rect_mat;
    std::vector<Eigen::MatrixXd> laneline_mat;

    EnvCrossroads(double size = 25.0, double width = 4.0);
    ~EnvCrossroads() {}
    void draw_env(void);
};


#endif
