#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include "instance.h"
#include "L1.h"
#include "cutting_plane.h"
#include "omp.h"
#include "DCA.h"

using namespace chrono;

int main() {

    string file_name = "record_500.csv";
    ofstream of;
    of.open(file_name);
    of << "method,nnz,running_time,size,p_coef,p_nnz" << endl;

    vector<string> algorithm_names = {"l1", "OMP", "Cap", "PiL", "SCAD", "CP", "ICP"};
    vector<double> running_time(algorithm_names.size(), 0);
    vector<double> nnz(algorithm_names.size(), -1);

//    vector<double> n_list{0.6, 1, 1.4};
//    vector<double> p_coef_list{0.2, 0.4, 0.6};
//    vector<double> p_nnz_list{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};

    vector<double> n_list{1.2};
    vector<double> p_coef_list{0.6};
    vector<double> p_nnz_list{0.3};
    int m = 1500;
    for (auto &s:n_list) {
        int n = static_cast<int>(m * s);
        for (auto &p_coef:p_coef_list) {
            for (auto &p_nnz:p_nnz_list) {
                for (int iter = 0; iter < 1; ++iter) {
                    cout << m << "," << n << "," << p_coef << "," << p_nnz << endl;
                    Instance inst(m, n);
                    system_clock::time_point t1 = high_resolution_clock::now();
                    generate_instance(inst, p_coef, p_nnz);
                    system_clock::time_point t2 = high_resolution_clock::now();
                    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
                    cout << "生成随机实例用时： " << time_span.count() << " s" << endl;


                    VectorXd x0 = VectorXd::Zero(n, 1);
                    t1 = high_resolution_clock::now();
                    l1_method(inst);
                    t2 = high_resolution_clock::now();
                    time_span = duration_cast<duration<double>>(t2 - t1);
                    running_time[0] = time_span.count();
                    if (inst.check_feasiblity()) {
                        nnz[0] = inst.count_nnz();
                        x0 = inst.x;
                    }

                    t1 = high_resolution_clock::now();
                    omp(inst);
                    t2 = high_resolution_clock::now();
                    time_span = duration_cast<duration<double>>(t2 - t1);
                    running_time[1] = time_span.count();
                    if (inst.check_feasiblity()) {
                        nnz[1] = inst.count_nnz();
                    }


                    inst.initialize_solution(x0);
                    t1 = high_resolution_clock::now();
                    DCA_Cap(inst);
                    t2 = high_resolution_clock::now();
                    time_span = duration_cast<duration<double>>(t2 - t1);
                    running_time[2] = time_span.count();
                    if (inst.check_feasiblity()) {
                        nnz[2] = inst.count_nnz();
                    }

                    inst.initialize_solution(x0);
                    t1 = high_resolution_clock::now();
                    DCA_PiL(inst);
                    t2 = high_resolution_clock::now();
                    time_span = duration_cast<duration<double>>(t2 - t1);
                    running_time[3] = time_span.count();
                    if (inst.check_feasiblity()) {
                        nnz[3] = inst.count_nnz();
                    }

                    inst.initialize_solution(x0);
                    t1 = high_resolution_clock::now();
                    DCA_SCAD(inst);
                    t2 = high_resolution_clock::now();
                    time_span = duration_cast<duration<double>>(t2 - t1);
                    running_time[4] = time_span.count();
                    if (inst.check_feasiblity()) {
                        nnz[4] = inst.count_nnz();
                    }


                    t1 = high_resolution_clock::now();
                    cutting_plane_method(inst);
                    t2 = high_resolution_clock::now();
                    time_span = duration_cast<duration<double>>(t2 - t1);
                    running_time[5] = time_span.count();
                    if (inst.check_feasiblity()) {
                        nnz[5] = inst.count_nnz();
                    }

                    t1 = high_resolution_clock::now();
                    ICP_method(inst, 0.4);
                    t2 = high_resolution_clock::now();
                    time_span = duration_cast<duration<double>>(t2 - t1);
                    running_time[6] = time_span.count();
                    if (inst.check_feasiblity()) {
                        nnz[6] = inst.count_nnz();
                    }

                    cout << "iterations: " << iter << "////////////////////////////" << endl;
                    for (int i = 0; i < algorithm_names.size(); ++i) {
                        of << algorithm_names[i] << ",";
                        of << nnz[i] << ",";
                        of << running_time[i] << ",";
                        of << m << "-" << n << ",";
                        of << p_coef << ",";
                        of << p_nnz << endl;

                        cout << algorithm_names[i] << " -- ";
                        cout << "非零变量个数：" << nnz[i] << ", ";
                        cout << "运行时间： " << running_time[i] << endl;
                    }
                }
            }
        }
    }


    return 0;
}