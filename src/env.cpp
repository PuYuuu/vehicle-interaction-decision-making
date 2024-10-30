/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-05-17 23:16:38
 * @LastEditTime: 2024-10-31 00:59:02
 * @FilePath: /vehicle-interaction-decision-making/src/env.cpp
 * Copyright 2024 puyu, All Rights Reserved.
 */

#include "matplotlibcpp.h"

#include "env.hpp"

namespace plt = matplotlibcpp;

EnvCrossroads::EnvCrossroads(double size, double width) :
    map_size(size), lanewidth(width)
{
   rect = {
        {{-size, -size, -2*lanewidth, -lanewidth, -lanewidth, -size},
            {-size, -lanewidth, -lanewidth, -2*lanewidth, -size, -size}},
        {{size, size, 2*lanewidth, lanewidth, lanewidth, size},
            {-size, -lanewidth, -lanewidth, -2*lanewidth, -size, -size}},
        {{size, size, 2*lanewidth, lanewidth, lanewidth, size},
            {size, lanewidth, lanewidth, 2*lanewidth, size, size}},
        {{-size, -size, -2*lanewidth, -lanewidth, -lanewidth, -size},
            {size, lanewidth, lanewidth, 2*lanewidth, size, size}}
    };

    laneline = {
        {{0, 0}, {-size, -2*lanewidth}},
        {{0, 0}, {size, 2*lanewidth}},
        {{-size, -2*lanewidth}, {0, 0}},
        {{size, 2*lanewidth}, {0, 0}}
    };

    for (const std::vector<std::vector<double>>& r : rect) {
        Eigen::MatrixXd mat(r.size(), r[0].size());
        for (int i = 0; i < r.size(); ++i) {
            for (int j = 0; j < r[0].size(); ++j) {
                mat(i, j) = r[i][j];
            }
        }
        rect_mat.emplace_back(mat);
    }
    
    for (const std::vector<std::vector<double>>& l : laneline) {
        Eigen::MatrixXd mat(l.size(), l[0].size());
        for (int i = 0; i < l.size(); ++i) {
            for (int j = 0; j < l[0].size(); ++j) {
                mat(i, j) = l[i][j];
            }
        }
        laneline_mat.emplace_back(mat);
    }
}

void EnvCrossroads::draw_env(void)
{
    plt::fill(rect[0][0], rect[0][1], {{"color", "#BFBFBF"}});
    plt::fill(rect[1][0], rect[1][1], {{"color", "#BFBFBF"}});
    plt::fill(rect[2][0], rect[2][1], {{"color", "#BFBFBF"}});
    plt::fill(rect[3][0], rect[3][1], {{"color", "#BFBFBF"}});
    plt::plot(rect[0][0], rect[0][1], {{"color", "k"}, {"linewidth", "2"}});
    plt::plot(rect[1][0], rect[1][1], {{"color", "k"}, {"linewidth", "2"}});
    plt::plot(rect[2][0], rect[2][1], {{"color", "k"}, {"linewidth", "2"}});
    plt::plot(rect[3][0], rect[3][1], {{"color", "k"}, {"linewidth", "2"}});

    plt::plot(laneline[0][0], laneline[0][1],
        {{"color", "orange"}, {"linewidth", "2"}, {"linestyle", "--"}});
    plt::plot(laneline[1][0], laneline[1][1],
        {{"color", "orange"}, {"linewidth", "2"}, {"linestyle", "--"}});
    plt::plot(laneline[2][0], laneline[2][1],
        {{"color", "orange"}, {"linewidth", "2"}, {"linestyle", "--"}});
    plt::plot(laneline[3][0], laneline[3][1],
        {{"color", "orange"}, {"linewidth", "2"}, {"linestyle", "--"}});
}
