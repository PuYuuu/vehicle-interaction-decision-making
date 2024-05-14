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
    std::vector<PredictTraj> predict_trajs;

    TrackedObject(std::string _name) : name(_name) {
        predict_trajs.clear();
    }
    ~TrackedObject() {
        
    }
};


#endif
