/*
 * @Author: puyu <yuu.pu@foxmail.com>
 * @Date: 2024-05-27 22:24:19
 * @LastEditTime: 2024-10-31 00:59:41
 * @FilePath: /vehicle-interaction-decision-making/src/imshow.cc
 * Copyright 2024 puyu, All Rights Reserved.
 */

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

#include <fmt/core.h>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using std::vector;

vector<float> imread(std::string filename, int& rows, int& cols, int& colors) {
    vector<float> image;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return image;
    }

    std::string line;
    getline(file, line);
    if (line != "Convert from PNG") {
        std::cerr << "不支持此格式: " << filename << std::endl;
        return image;
    }
    getline(file, line);
    std::istringstream iss(line);
    iss >> rows >> cols >> colors;
    image.resize(rows * cols * colors);
    int idx = 0;
    while (getline(file, line)) {
        std::istringstream iss(line);
        for (int i = 0; i < colors; ++i) {
            iss >> image[idx++];
        }
    }

    file.close();

    // directly return will trigger RVO (Return Value Optimization)
    return std::move(image);
}

int main(int argc, char *argv[]) {
    Py_Initialize();
    _import_array();

    std::filesystem::path source_file_path(__FILE__);
    std::filesystem::path project_path = source_file_path.parent_path().parent_path();
    std::string script_path = project_path / "scripts";
    int cols;
    int rows;
    int colors;
    vector<float> image =
        imread("/home/puyu/Codes/vehicle-interaction-decision-making/img/vehicle/blue.mat.txt",
        rows, cols, colors);

    PyRun_SimpleString("import sys");
    PyRun_SimpleString(fmt::format("sys.path.append('{}')", script_path).c_str());

    // 导入Python模块
    PyObject *pName = PyUnicode_DecodeFSDefault("imshow");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "imshow");

        if (pFunc && PyCallable_Check(pFunc)) {
            std::vector<double> x{0, 0, 1.57};
            std::vector<double> y{5, 2};

            PyObject* state = matplotlibcpp::detail::get_array(x);
            PyObject* vehicle_para = matplotlibcpp::detail::get_array(y);
            npy_intp dims[3] = { rows, cols, colors };

            const float* imptr = &(image[0]);

            PyObject *args = PyTuple_New(3);
            PyTuple_SetItem(args, 0, PyArray_SimpleNewFromData(3, dims, NPY_FLOAT, (void *)imptr));
            PyTuple_SetItem(args, 1, state);
            PyTuple_SetItem(args, 2, vehicle_para);

            PyObject *pValue = PyObject_CallObject(pFunc, args);

            Py_DECREF(args);

            if (pValue != nullptr) {
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
                fprintf(stderr, "Call failed\n");
            }
            Py_DECREF(pFunc);
        } else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"my_function\"\n");
        }
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"script\"\n");
    }

    Py_Finalize();
    return 0;
}
