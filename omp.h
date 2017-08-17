//
// Created by agamemnon on 17-8-16.
//

#ifndef RESOURCE_REALLOCATION_OMP_H
#define RESOURCE_REALLOCATION_OMP_H

#include <ilcplex/ilocplex.h>
#include <iostream>
#include <set>
#include <map>
#include "instance.h"


ILOSTLBEGIN


void omp(Instance &inst) {
    MatrixXd &A = inst.A;
    VectorXd &l = inst.l;
    VectorXd &u = inst.u;
    const int m = A.rows();
    const int n = A.cols();
    VectorXd c(2 * m);
    c << -l, u;
    MatrixXd M(2 * m, n);
    M << -A, A;
//    cout<<M<<endl;
    set<int> res_cols, selected_cols;
    for (int j = 0; j < n; ++j) {
        res_cols.insert(j);
    }
    map<int, double> projection;
    for (int j = 0; j < n; ++j) {
        projection[j] = 0.0;
    }
    int iter = 0;
    while ((c.norm() > 1e-4) && (iter < 5)) {
        double max_projection = 0;
        int select_j = *res_cols.begin();
        for (auto &j:res_cols) {
            projection[j] = abs(c.dot(M.col(j)) / M.col(j).norm());
            if (projection.at(j) > max_projection) {
                max_projection = projection.at(j);
                select_j = j;
            }
            cout << projection.at(j) << endl;
        c = c - c.dot(M.col(select_j)) / M.col(select_j).norm();
        cout << iter << c.norm() << endl;
        iter += 1;
    }


}

#endif //RESOURCE_REALLOCATION_OMP_H
