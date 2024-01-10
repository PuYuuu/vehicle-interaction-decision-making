#include "env.hpp"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main(int argc, char** argv)
{
    EnvCrossroads env(25, 4);
    env.draw_env();

    plt::xlim(-25.0, 25.0);
    plt::ylim(-25.0, 25.0);
    plt::set_aspect_equal();
    plt::show();

    return 0;
}