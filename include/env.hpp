#pragma once
#ifndef __ENV_HPP
#define __ENV_HPP

#include <vector>


class EnvCrossroads
{
private:

public:
    double map_size;
    double lanewidth;

    std::vector<std::vector<std::vector<double>>> rect;
    std::vector<std::vector<std::vector<double>>> laneline;

    EnvCrossroads(double size = 25.0, double width = 4.0);
    ~EnvCrossroads() {}
    void draw_env(void);
};


#endif
