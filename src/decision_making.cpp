#include <cmath>
#include <random>
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
#include "planner.hpp"

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

void run(int rounds_num, string config_path,
    string save_path, bool show_animation, bool save_fig) {
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const YAML::Exception& e) {
        spdlog::error(fmt::format("Error parsing YAML file: {}", e.what()));
        return ;
    }

    // initialize
    double delta_t = config["delta_t"].as<double>();
    std::shared_ptr<EnvCrossroads> env = std::make_shared<EnvCrossroads>(25, 4);
    VehicleBase::initialize(env, 5, 2, 8, 2.4);
    MonteCarloTreeSearch::initialize(config);
    Node::initialize(config["max_step"].as<int>(), MonteCarloTreeSearch::calc_cur_value);

    std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<double> random_pos_0(-20, -12);
    std::uniform_real_distribution<double> random_pos_1(12, 20);
    std::uniform_real_distribution<double> random_velo(3, 5);
    uint64_t max_per_iters = static_cast<uint64_t>(20 / delta_t);

    uint64_t succeed_count = 0;
    for (uint64_t iter = 0; iter < rounds_num; ++iter) {
        double init_y_0 = random_pos_0(engine);
        double init_y_1 = random_pos_1(engine);
        double init_v_0 = random_velo(engine);
        double init_v_1 = random_velo(engine);

        Vehicle vehicle_0 = \
                Vehicle("vehicle_0", State(env->lanewidth/2, init_y_0, M_PI_2, init_v_0), "blue", config);
        Vehicle vehicle_1 = \
                Vehicle("vehicle_1", State(-env->lanewidth/2, init_y_1, -M_PI_2, init_v_1), "red", config);

        vehicle_0.set_level(1);
        vehicle_1.set_level(0);
        vehicle_0.set_target(State(-18, env->lanewidth / 2, M_PI, 0));
        vehicle_1.set_target(State(-env->lanewidth / 2, -18, 1.5 * M_PI, 0));
        std::vector<State> vehicle_0_history = {vehicle_0.state};
        std::vector<State> vehicle_1_history = {vehicle_1.state};

        spdlog::info(fmt::format("================== Round {} ==================", iter));

        uint64_t cur_loop_count = 0;
        while (true) {
            if (vehicle_1.is_get_target()) {
                spdlog::info(fmt::format("Round {} successed !", iter));
                break;
            }

            if (has_overlap(VehicleBase::get_box2d(vehicle_0.state), VehicleBase::get_box2d(vehicle_1.state)) ||
                cur_loop_count > max_per_iters) {
                spdlog::info(fmt::format("Round {} failed !", iter));
                break;
            }

            // auto act_and_traj_0 = vehicle_0.excute(vehicle_1);
            auto act_and_traj_1 = vehicle_1.excute(vehicle_0);

            // vehicle_0.state = kinematic_propagate(
            //                     vehicle_0.state, get_action_value(act_and_traj_0.first), delta_t);
            vehicle_1.state = kinematic_propagate(
                                vehicle_1.state, get_action_value(act_and_traj_1.first), delta_t);
            StateList excepted_traj_1 = act_and_traj_1.second;
            std::vector<std::vector<double>> traj_vec_1 = excepted_traj_1.to_vector();

            if (show_animation) {
                plt::cla();
                env->draw_env();
                vehicle_0.draw_vehicle();
                vehicle_1.draw_vehicle();
                plt::plot({vehicle_0.target.x}, {vehicle_0.target.y}, {{"marker", "x"}, {"color", vehicle_0.color}});
                plt::plot({vehicle_1.target.x}, {vehicle_1.target.y}, {{"marker", "x"}, {"color", vehicle_1.color}});
                plt::plot(traj_vec_1[0], traj_vec_1[1], {{"color", vehicle_1.color}, {"linewidth", "1"}});
                plt::text(10, -15, fmt::format("v = {:.2f} m/s", vehicle_0.state.v));
                plt::text(10,  15, fmt::format("v = {:.2f} m/s", vehicle_1.state.v));
                // plt::text(10, -18, fmt::format("{}", get_action_name(act_and_traj_0.first)));
                plt::text(10,  12, fmt::format("{}", get_action_name(act_and_traj_1.first)));
                plt::xlim(-25.0, 25.0);
                plt::ylim(-25.0, 25.0);
                plt::set_aspect_equal();
                plt::pause(1);
            }
        }
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
