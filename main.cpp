#include <iostream>
#include <chrono>
#include "instance.h"
#include "L1.h"
#include "cutting_plane.h"
#include "omp.h"
#include "DCA.h"

using namespace chrono;

int main() {
    std::cout << "Hello, World!" << std::endl;
    Instance inst(500, 540);
//    inst.display();
    system_clock::time_point t1 = high_resolution_clock::now();
    generate_instance(inst, 0.4, 0.25);
    system_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << "生成随机实例用时： " << time_span.count() << " s" << endl;


    int nnz1 = -1, nnz2 = -1, nnz3 = -1, nnz4 = -1;
    t1 = high_resolution_clock::now();
    DCA_Cap(inst, 0.05, 1e-4);
    if (inst.check_feasiblity()) {
        nnz1 = inst.count_nnz();
    } else {
        nnz1 = -1;
    }
    inst.clear_solution();

    l1_method(inst);
    if (inst.check_feasiblity()) {
        nnz2 = inst.count_nnz();
    } else {
        nnz2 = -1;
    }
    inst.clear_solution();
    DCA_SCAD(inst);
    if (inst.check_feasiblity()) {
        nnz3 = inst.count_nnz();
    } else {
        nnz3 = -1;
    }
    inst.clear_solution();

    DCA_PiL(inst);
    if (inst.check_feasiblity()) {
        nnz4 = inst.count_nnz();
    } else {
        nnz4 = -1;
    }
    inst.clear_solution();

    t2 = high_resolution_clock::now();


    cout << "非零变量个数：" << nnz1 << endl;
    cout << "非零变量个数：" << nnz2 << endl;
    cout << "非零变量个数：" << nnz3 << endl;
    cout << "非零变量个数：" << nnz4 << endl;

    time_span = duration_cast<duration<double>>(t2 - t1);
    cout << "求解用时： " << time_span.count() << " s" << endl;


    return 0;
}