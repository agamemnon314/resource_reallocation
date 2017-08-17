#include <iostream>
#include <chrono>
#include "instance.h"
#include "L1.h"
#include "cutting_plane.h"
//#include "omp.h"

using namespace chrono;

int main() {
    std::cout << "Hello, World!" << std::endl;
    Instance inst(300, 340);
//    inst.display();
    system_clock::time_point t1 = high_resolution_clock::now();
    generate_instance(inst, 0.4, 0.25);
    system_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << "生成随机实例用时： " << time_span.count() << " s" << endl;

    t1 = high_resolution_clock::now();
    l1_method(inst);
    cutting_plane_method(inst);
//    omp(inst);
    t2 = high_resolution_clock::now();

    time_span = duration_cast<duration<double>>(t2 - t1);
    cout << "求解用时： " << time_span.count() << " s" << endl;


    return 0;
}