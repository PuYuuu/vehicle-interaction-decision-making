#include <matplotlib-cpp/matplotlibcpp.h>

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
