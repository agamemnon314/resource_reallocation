//
// Created by agamemnon on 17-8-16.
//

#ifndef RESOURCE_REALLOCATION_INSTANCE_H
#define RESOURCE_REALLOCATION_INSTANCE_H

#include <iostream>
#include <Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;

struct Instance {
    MatrixXd A;
    VectorXd l;
    VectorXd u;

    Instance(const int m, const int n) {
        A = MatrixXd::Zero(m, n);
        l = VectorXd::Zero(m);
        u = VectorXd::Zero(m);
    }

    void display() {
        cout << "A" << endl;
        cout << A << endl;
        cout << "l" << endl;
        cout << l.transpose() << endl;
        cout << "u" << endl;
        cout << u.transpose() << endl;
    }
};

void generate_instance(Instance &inst, double p, double alpha) {
    random_device rd;
    default_random_engine dre(rd());
    bernoulli_distribution bd(p);
    int m = inst.A.rows();
    int n = inst.A.cols();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            inst.A(i, j) = bd(dre);
        }
    }

    VectorXd x = VectorXd::Zero(n);
    int nnz = 0;
    uniform_int_distribution<int> uid(0, n - 1);
    normal_distribution<double> nd;
    while (nnz < m * alpha) {
        int j = uid(dre);
        if (x(j) == 0) {
            x(j) = 50 * nd(dre);
            nnz += 1;
        }
    }

    inst.l = inst.A * x;
    inst.u = inst.A * x;
    for (int i = 0; i < m; ++i) {
        inst.l(i) -= 100 * abs(nd(dre));
        inst.u(i) += 100 * abs(nd(dre));
    }
}

#endif //RESOURCE_REALLOCATION_INSTANCE_H
