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

    MatrixXd rays = MatrixXd::Identity(2 * m, 2 * m);
    MatrixXd projected_rays = rays;
    VectorXd cR = c.transpose() * rays;


    int iter = 0;
    while ((cR.minCoeff() < 0) && (iter < 500)) {
        VectorXd unbounded_direction = VectorXd::Zero(2 * m, 1);
        for (int i = 0; i < 2 * m; ++i) {
            if (cR[i] < 0) {
                unbounded_direction += projected_rays.col(i);
            }
        }

        double most_violated_value = 0;
        int most_violated_cutting_plane_id = -1;
        double violated_value;
        for (auto &j:res_cols) {
            violated_value = abs(M.col(j).dot(unbounded_direction));
            if (violated_value > most_violated_value) {
                most_violated_value = violated_value;
                most_violated_cutting_plane_id = j;
            }
        }


//        res_cols.erase(most_violated_cutting_plane_id);
        selected_cols.insert(most_violated_cutting_plane_id);

        MatrixXd P(2 * m, res_cols.size());
        int j_P = 0;
        for (auto &j:res_cols) {
            P.col(j_P) << M.col(j);
            j_P++;
        }
        projected_rays = rays - M * (M.transpose() * M).inverse() * M.transpose();
        cR = c.transpose() * projected_rays;

        cout << cR.minCoeff() << endl;
        cout << M * (M.transpose() * M).inverse() * M.transpose() << endl;

        iter += 1;
    }


}

#endif //RESOURCE_REALLOCATION_OMP_H
