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
        cplex.setOut(env.getNullStream());


        cplex.solve();
        if (cplex.getStatus() == IloAlgorithm::Infeasible) {
            cout << "当前问题不可行" << endl;
        }
        if (cplex.getStatus() == IloAlgorithm::Optimal) {
            for (int j = 0; j < n; ++j) {
                inst.x[j] = cplex.getValue(x[j]);
            }
//            int nnz = 0;
//            for (int j = 0; j < n; ++j) {
//                if (abs(cplex.getValue(x[j])) > 1e-4) {
//                    nnz += 1;
//                }
//            }
//            cout << "非零变量个数：" << nnz << endl;
        }
    } catch (const IloException &e) {
        cerr << "Exception caught: " << e << endl;
    } catch (...) {
        cerr << "Unknown exception caught!" << endl;
    }

    env.end();
}

void reweighted_l1(Instance &inst, double epsilon = 0.1) {
    MatrixXd &A = inst.A;
    VectorXd &l = inst.l;
    VectorXd &u = inst.u;
    const int m = A.rows();
    const int n = A.cols();

    VectorXd x_cur = inst.x;
    VectorXd x_next = VectorXd::Zero(n, 1);
    VectorXd y = VectorXd::Zero(n, 1);

    IloEnv env;
    try {

        IloModel model(env);
        IloNumVarArray x(env, n, -IloInfinity, IloInfinity, ILOFLOAT);
        IloNumVarArray t(env, n, 0, IloInfinity, ILOFLOAT);
        vector<double> w(n, 1.0);


        IloExpr obj_expr(env);
        IloObjective obj(env);
        for (int j = 0; j < n; ++j) {
            obj_expr += w[j] * t[j];
        }
        obj = IloMinimize(env, obj_expr);
        model.add(obj);
        obj_expr.clear();


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
            model.add(t[j] >= x[j]);
            model.add(-t[j] <= x[j]);
        }

        expr.end();

        IloCplex cplex(model);
        cplex.setOut(env.getNullStream());

        cplex.solve();
        if (cplex.getStatus() == IloAlgorithm::Infeasible) {
            cout << "当前问题不可行" << endl;
        }
        bool is_feasible = (cplex.getStatus() == IloAlgorithm::Optimal)
                           || (cplex.getStatus() == IloAlgorithm::Feasible);
        double step_size = 10;
        while (is_feasible && step_size > 1e-3) {
//            cout << step_size << endl;
            for (int j = 0; j < n; ++j) {
                x_next[j] = cplex.getValue(x[j]);
            }
            step_size = (x_next - x_cur).norm() / (x_cur.norm() + 1);
            x_cur = x_next;

            for (int j = 0; j < n; ++j) {
                w[j] = 1 / (epsilon + abs(x_next[j]));
            }
            model.remove(obj);
            for (int j = 0; j < n; ++j) {
                obj_expr += w[j] * t[j];
            }
            obj = IloMinimize(env, obj_expr);
            model.add(obj);
            obj_expr.clear();
            cplex.solve();
            is_feasible = (cplex.getStatus() == IloAlgorithm::Optimal)
                          || (cplex.getStatus() == IloAlgorithm::Feasible);


        }
        if (cplex.getStatus() == IloAlgorithm::Infeasible) {
            cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        }

        if (is_feasible) {
            for (int j = 0; j < n; ++j) {
                inst.x[j] = cplex.getValue(x[j]);
            }
        }
    } catch (const IloException &e) {
        cerr << "Exception caught: " << e << endl;
    } catch (string str) {
        cerr << str << " is not avaiable!" << endl;
    } catch (...) {
        cerr << "Unknown exception caught!" << endl;
    }

    env.end();
}


#endif //RESOURCE_REALLOCATION_L1_H
