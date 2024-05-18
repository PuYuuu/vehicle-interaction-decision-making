#pragma once
#ifndef __TRACED_OBJECT_HPP
#define __TRACED_OBJECT_HPP

#include <string>
#include <vector>

#include "utils.hpp"

struct PredictTraj {
    double confidence;
    StateList traj; 
};


class TrackedObject {
private:
    /* data */
public:
    std::string name;
    State state;
    State target;
    std::vector<PredictTraj> predict_trajs;

    TrackedObject() : name("") {
        state = State(0, 0, 0, 0);
        predict_trajs.clear();
    }
    TrackedObject(std::string _name) : name(_name) {
        state = State(0, 0, 0, 0);
        predict_trajs.clear();
    }
    ~TrackedObject() {

    }
};


#endif
