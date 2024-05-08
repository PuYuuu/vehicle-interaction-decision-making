#include <cmath>
#include <random>
#include <string>
#include <memory>
#include <getopt.h>
#include <filesystem>
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
    {"no_animation", no_argument, 0, 'n'},
    {"save_fig", no_argument, 0, 'f'},
};

std::unordered_map<int, spdlog::level::level_enum> LOG_LEVEL_DICT =
    {{0, spdlog::level::trace}, {1, spdlog::level::debug}, {2, spdlog::level::info},
     {3, spdlog::level::warn}, {4, spdlog::level::err}, {5, spdlog::level::critical}};

void run(int rounds_num, std::filesystem::path config_path,
    std::filesystem::path save_path, bool show_animation, bool save_fig) {
    YAML::Node config;
    spdlog::info(fmt::format("config path: {}", config_path.string()));
    try {
        config = YAML::LoadFile(config_path.string());
        spdlog::info(fmt::format("config parameters:\n{}", YAML::Dump(config)));
    } catch (const YAML::Exception& e) {
        spdlog::error(fmt::format("Error parsing YAML file: {}", e.what()));
        return ;
    }

    // initialize
    double delta_t = config["delta_t"].as<double>();
    double max_simulation_time = config["max_simulation_time"].as<double>();
    std::shared_ptr<EnvCrossroads> env = std::make_shared<EnvCrossroads>(25, 4);
    VehicleBase::initialize(env, 5, 2, 8, 2.4);
    MonteCarloTreeSearch::initialize(config);
    Node::initialize(config["max_step"].as<int>(), MonteCarloTreeSearch::calc_cur_value);

    std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<double> random_pos_0(-20, -12);
    std::uniform_real_distribution<double> random_pos_1(12, 20);
    std::uniform_real_distribution<double> random_velo(3, 5);

    uint64_t succeed_count = 0;
    for (uint64_t iter = 0; iter < rounds_num; ++iter) {
        double init_y_0 = random_pos_0(engine);
        double init_y_1 = random_pos_1(engine);
        double init_v_0 = random_velo(engine);
        double init_v_1 = random_velo(engine);

        std::shared_ptr<Vehicle> vehicle_0 = std::make_shared<Vehicle>(
            "vehicle_0", State(env->lanewidth/2, init_y_0, M_PI_2, init_v_0), "blue", config);
        std::shared_ptr<Vehicle> vehicle_1 = std::make_shared<Vehicle>(
            "vehicle_1", State(-env->lanewidth/2, init_y_1, -M_PI_2, init_v_1), "red", config);

        vehicle_0->set_level(1);
        vehicle_1->set_level(0);
        vehicle_0->set_target(State(-18, env->lanewidth / 2, M_PI, 0));
        vehicle_1->set_target(State(-env->lanewidth / 2, -18, 1.5 * M_PI, 0));
        
        VehicleList vehicles({vehicle_0, vehicle_1});

        std::vector<State> vehicle_0_history = {vehicles[0]->state};
        std::vector<State> vehicle_1_history = {vehicles[1]->state};

        spdlog::info(fmt::format("================== Round {} ==================", iter));
        for (auto vehicle : vehicles) {
            spdlog::info(fmt::format("{} >>> init_x: {:.2f}, init_y: {:.2f}, init_v: {:.2f}",
                                vehicle->name, vehicle->state.x, vehicle->state.y, vehicle->state.v));
        }

        double timestamp = 0.0;
        TicToc total_cost_time;
        while (true) {
            if (vehicles.is_all_get_target()) {
                spdlog::info(fmt::format(
                        "Round {:d} successed, simulation time: {:.3f} s, actual timecost: {:.3f} s",
                        iter, timestamp, total_cost_time.toc()));
                ++succeed_count;
                break;
            }

            if ( vehicles.is_any_collision() || timestamp > max_simulation_time) {
                spdlog::info(fmt::format(
                        "Round {:d} failed, simulation time: {.3f} s, actual timecost: {.3f} s",
                        iter, timestamp, total_cost_time.toc()));
                break;
            }

            TicToc iter_cost_time;
            auto act_and_traj_0 = vehicle_0->excute(*vehicle_1);
            auto act_and_traj_1 = vehicle_1->excute(*vehicle_0);

            vehicle_0->state = utils::kinematic_propagate(
                                vehicle_0->state, utils::get_action_value(act_and_traj_0.first), delta_t);
            vehicle_1->state = utils::kinematic_propagate(
                                vehicle_1->state, utils::get_action_value(act_and_traj_1.first), delta_t);
            vehicle_0_history.push_back(vehicle_0->state);
            vehicle_1_history.push_back(vehicle_1->state);

            spdlog::debug(fmt::format("single step cost {:.3f} sec", iter_cost_time.toc()));            
            
            StateList excepted_traj_1 = act_and_traj_1.second;
            StateList excepted_traj_0 = act_and_traj_0.second;
            auto traj_vec_1 = excepted_traj_1.to_vector();
            auto traj_vec_0 = excepted_traj_0.to_vector();

            if (show_animation) {
                plt::cla();
                env->draw_env();
                for (auto vehicle : vehicles) {
                    vehicle->draw_vehicle();
                    plt::plot({vehicle->target.x}, {vehicle->target.y}, {{"marker", "x"}, {"color", vehicle->color}});
                }
                plt::plot(traj_vec_0[0], traj_vec_0[1], {{"color", vehicle_0->color}, {"linewidth", "1"}});
                plt::plot(traj_vec_1[0], traj_vec_1[1], {{"color", vehicle_1->color}, {"linewidth", "1"}});
                plt::text(10, -15, fmt::format("v = {:.2f} m/s", vehicle_0->state.v), {{"color", vehicle_0->color}});
                plt::text(10,  15, fmt::format("v = {:.2f} m/s", vehicle_1->state.v), {{"color", vehicle_1->color}});
                plt::text(10, -18, fmt::format("{}",
                            utils::get_action_name(act_and_traj_0.first)), {{"color", vehicle_0->color}});
                plt::text(10,  12, fmt::format("{}",
                            utils::get_action_name(act_and_traj_1.first)), {{"color", vehicle_1->color}});
                plt::xlim(-25.0, 25.0);
                plt::ylim(-25.0, 25.0);
                plt::set_aspect_equal();
                plt::pause(1);
            }
            timestamp += delta_t;
        }

        plt::clf();
        env->draw_env();
        for (auto history : vehicle_0_history) {
            Vehicle tmp("tmp", history, "blue", config);
            tmp.draw_vehicle(true);
        }
        for (auto history : vehicle_1_history) {
            Vehicle tmp("tmp", history, "red", config);
            tmp.draw_vehicle(true);
        }
        plt::xlim(-25, 25);
        plt::ylim(-25, 25);
        plt::set_aspect_equal();
        if (save_fig) {
            plt::save((save_path/(std::to_string(iter)+".svg")).string(), 600);
        }
    }

    double succeed_rate = 100 * succeed_count / rounds_num;
    spdlog::info("\n=========================================");
    spdlog::info(fmt::format("Experiment success {}/{}({:.2f}%) rounds.", succeed_count, rounds_num, succeed_rate));
}

int main(int argc, char** argv) {
    std::filesystem::path source_file_path(__FILE__);
    std::filesystem::path project_path = source_file_path.parent_path().parent_path();

    int rounds_num = 5;
    std::filesystem::path output_path = project_path / "logs";
    std::filesystem::path config_path = "default.yaml";
    bool show_animation = true;
    bool save_flag = false;
    int log_level = 2;      // info

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "r:o:l:c:n:f:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'r':
                rounds_num = std::stoi(optarg);
                break;
            case 'o':
                output_path = utils::absolute_path(optarg);
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

    // config file path
    config_path = project_path / "config" / config_path;

    // output path
    if (save_flag) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
        struct tm now_tm;
        localtime_r(&now_time_t, &now_tm); 
        std::stringstream ss;
        ss << std::put_time(&now_tm, "%Y-%m-%d-%H-%M-%S");
        output_path = output_path / ss.str();
        if (!std::filesystem::exists(output_path)) {
            std::filesystem::create_directories(output_path);
        }
    }

    run(rounds_num, config_path, output_path, show_animation, save_flag);

    return 0;
}
