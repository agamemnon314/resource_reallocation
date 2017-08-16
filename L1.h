//
// Created by agamemnon on 17-8-16.
//

#ifndef RESOURCE_REALLOCATION_L1_H
#define RESOURCE_REALLOCATION_L1_H

#include <ilcplex/ilocplex.h>
#include <iostream>
#include "instance.h"


ILOSTLBEGIN


void l1_method(Instance &inst) {
    MatrixXd &A = inst.A;
    VectorXd &l = inst.l;
    VectorXd &u = inst.u;
    const int m = A.rows();
    const int n = A.cols();
    IloEnv env;
    try {

        IloModel model(env);
        IloNumVarArray x(env, n, -IloInfinity, IloInfinity, ILOFLOAT);
        IloNumVarArray y(env, n, 0, IloInfinity, ILOFLOAT);

        IloExpr l1_norm(env);
        for (int j = 0; j < n; ++j) {
            l1_norm += y[j];
        }
        model.add(IloMinimize(env, l1_norm));


        IloExpr expr(env);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                expr += A(i, j) * x[j];
            }
            model.add(expr <= u(i));
            model.add(expr >= l(i));
            expr.clear();
        }

        for (int j = 0; j < n; ++j) {
            model.add(y[j] >= x[j]);
            model.add(-y[j] <= x[j]);
        }

        expr.end();

        IloCplex cplex(model);

        cplex.solve();
        if (cplex.getStatus() == IloAlgorithm::Infeasible) {
            cout << "当前问题不可行" << endl;
        }
        if (cplex.getStatus() == IloAlgorithm::Optimal) {
            int nnz = 0;
            for (int j = 0; j < n; ++j) {
                if (abs(cplex.getValue(x[j])) > 1e-4) {
                    nnz += 1;
                }
            }
            cout << "非零变量个数：" << nnz << endl;
        }
    } catch (const IloException &e) {
        cerr << "Exception caught: " << e << endl;
    } catch (...) {
        cerr << "Unknown exception caught!" << endl;
    }

    env.end();
}


#endif //RESOURCE_REALLOCATION_L1_H
