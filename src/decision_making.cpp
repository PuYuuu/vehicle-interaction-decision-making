#include <cmath>
#include <string>
#include <memory>
#include <getopt.h>
#include <unordered_map>

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include <matplotlib-cpp/matplotlibcpp.h>

#include "env.hpp"
#include "utils.hpp"
#include "vehicle.hpp"
#include "vehicle_base.hpp"

using std::string;
namespace plt = matplotlibcpp;

static struct option long_options[] = {
    {"rounds", required_argument, 0, 'r'},
    {"output_path", required_argument, 0, 'o'},
    {"log_level", required_argument, 0, 'l'},
    {"config", required_argument, 0, 'c'},
    {"no_animation", required_argument, 0, 'n'},
    {"save_fig", required_argument, 0, 'f'},
};

std::unordered_map<int, spdlog::level::level_enum> LOG_LEVEL_DICT =
    {{0, spdlog::level::trace}, {1, spdlog::level::debug}, {0, spdlog::level::info},
     {0, spdlog::level::warn}, {0, spdlog::level::err}, {0, spdlog::level::critical}};

void run(int rounds_num, string config_path, string save_path, bool show_animation, bool save_fig) {
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        spdlog::error(fmt::format("Error parsing YAML file: {}", e.what()));
        return ;
    }

    double delta_t = config["delta_t"].as<double>();
    std::shared_ptr<EnvCrossroads> env = std::make_shared<EnvCrossroads>(25, 4);
    VehicleBase::initialize(env, 5, 2, 8, 2.4);

    Vehicle vehicle_0 = \
            Vehicle("vehicle_0", State(env->lanewidth/2, -15, M_PI_2, 0), "blue", config);

    if (show_animation) {
        plt::cla();
        env->draw_env();
        vehicle_0.draw_vehicle();

        plt::xlim(-25.0, 25.0);
        plt::ylim(-25.0, 25.0);
        plt::set_aspect_equal();
        plt::show();
    }

}

int main(int argc, char** argv) {
    int rounds_num = 5;
    string output_path;
    string config_path;
    bool show_animation = true;
    bool save_flag = false;
    int log_level = 1;      // debug

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "r:o:l:c:n:f:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'r':
                rounds_num = std::stoi(optarg);
                break;
            case 'o':
                output_path = optarg;
                break;
            case 'l':
                log_level = std::stoi(optarg);
                break;
            case 'c':
                config_path = optarg;
                break;
            case 'n':
                show_animation = false;
                break;
            case 'f':
                save_flag = true;
                break;
            default:
                exit(EXIT_FAILURE);
        }
    }

    spdlog::set_level(LOG_LEVEL_DICT[log_level]);
    spdlog::set_pattern("%Y-%m-%d %H:%M:%S.%e - %^%l%$ - %v");

    config_path = "/home/puyu/Codes/vehicle-interaction-decision-making/config/default.yaml";

    run(rounds_num, config_path, output_path, show_animation, save_flag);

    return 0;
}
