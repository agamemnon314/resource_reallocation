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

void calculate_residual(VectorXd &res, MatrixXd &A, VectorXd &x, VectorXd &l, VectorXd &u) {
    VectorXd delta_l = -l + (A * x);
    VectorXd delta_u = -u + (A * x);
    int n = A.rows();
    for (int i = 0; i < n; ++i) {
        if (delta_l[i] < -1e-4) {
            res[i] = delta_l[i];
        }
        if (delta_u[i] > 1e-4) {
            res[i] = delta_u[i];
        }
    }
}


void omp(Instance &inst) {
    MatrixXd &A = inst.A;
    VectorXd &l = inst.l;
    VectorXd &u = inst.u;
    const int m = A.rows();
    const int n = A.cols();
    set<int> res_cols, selected_cols;
    for (int j = 0; j < n; ++j) {
        res_cols.insert(j);
    }

    VectorXd xx = VectorXd::Zero(n, 1);
    VectorXd res(m, 1);
    calculate_residual(res, A, xx, l, u);
    double res_dot_a;
    double max_res_dot_a = -1;
    int select_j = -1;
    for (auto &j:res_cols) {
        res_dot_a = abs(A.col(j).dot(res)) / A.col(j).norm();
        if (res_dot_a > max_res_dot_a) {
            max_res_dot_a = res_dot_a;
            select_j = j;
        }
    }
    selected_cols.insert(select_j);
    res_cols.erase(select_j);

    IloEnv env;
    try {

        IloModel model(env);
        IloNumVarArray x(env, n, -IloInfinity, IloInfinity, ILOFLOAT);
        IloNumVarArray y(env, m, 0, IloInfinity, ILOFLOAT);

        IloExpr obj(env);
        for (int i = 0; i < m; ++i) {
            obj += y[i];
        }
        model.add(IloMinimize(env, obj));


        IloExpr expr(env);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                expr += A(i, j) * x[j];
            }
            model.add(expr - u(i) <= y[i]);
            model.add(-expr + l(i) <= y[i]);
            expr.clear();
        }

        for (int j = 0; j < n; ++j) {
            if (selected_cols.find(j) == selected_cols.end()) {
                x[j].setBounds(0, 0);
            } else {
                x[j].setBounds(-IloInfinity, IloInfinity);
            }
        }

        expr.end();

        IloCplex cplex(model);
        cplex.setOut(env.getNullStream());
        cplex.solve();
        if (cplex.getStatus() == IloAlgorithm::Infeasible) {
            cout << "当前问题不可行" << endl;
            return;
        }
        bool is_feasible = (cplex.getStatus() == IloAlgorithm::Optimal)
                           || (cplex.getStatus() == IloAlgorithm::Feasible);
        while (is_feasible && (cplex.getObjValue() > 1e-4)) {
//            cout << cplex.getObjValue() << endl;
            xx = VectorXd::Zero(n, 1);
            for (auto &j:selected_cols) {
                xx[j] = cplex.getValue(x[j]);
            }
            calculate_residual(res, A, xx, l, u);
            max_res_dot_a = -1;
            select_j = -1;
            for (auto &j:res_cols) {
                res_dot_a = abs(A.col(j).dot(res)) / A.col(j).norm();
                if (res_dot_a > max_res_dot_a) {
                    max_res_dot_a = res_dot_a;
                    select_j = j;
                }
            }
            selected_cols.insert(select_j);
            res_cols.erase(select_j);
            for (int j = 0; j < n; ++j) {
                if (selected_cols.find(j) == selected_cols.end()) {
                    x[j].setBounds(0, 0);
                } else {
                    x[j].setBounds(-IloInfinity, IloInfinity);
                }
            }

            cplex.solve();
            is_feasible = (cplex.getStatus() == IloAlgorithm::Optimal)
                          || (cplex.getStatus() == IloAlgorithm::Feasible);
        }


        if (is_feasible && (cplex.getObjValue() <= 1e-4)) {
            xx = VectorXd::Zero(n, 1);
            for (auto &j:selected_cols) {
                xx[j] = cplex.getValue(x[j]);
            }
            inst.x = xx;
        }
    } catch (const IloException &e) {
        cerr << "Exception caught: " << e << endl;
    } catch (...) {
        cerr << "Unknown exception caught!" << endl;
    }

    env.end();

}

#endif //RESOURCE_REALLOCATION_OMP_H
