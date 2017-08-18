//
// Created by agamemnon on 17-8-17.
//

#ifndef RESOURCE_REALLOCATION_CUTTING_PLANE_H
#define RESOURCE_REALLOCATION_CUTTING_PLANE_H

#include <ilcplex/ilocplex.h>
#include <iostream>
#include <map>
#include <set>
#include <list>
#include "instance.h"


ILOSTLBEGIN

struct DualValue {
    int col;
    double dual_value;

    bool operator<(const DualValue &that) const {
        return dual_value < that.dual_value;
    }
};

void cutting_plane_method(Instance &inst) {
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
    IloEnv env;
    try {

        IloModel model(env);
        IloFloatVarArray y(env, 2 * m, 0, 1);

        IloExpr obj(env);
        for (int i = 0; i < 2 * m; ++i) {
            obj += c[i] * y[i];
        }
        model.add(IloMinimize(env, obj));

        IloRangeArray cutting_pool(env, n);

        IloExpr expr(env);

        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                expr += -A(i, j) * y[i];
                expr += A(i, j) * y[i + m];
            }
            cutting_pool[j] = expr == 0;
            expr.clear();
        }

        expr.end();

//        model.add(cutting_pool);

        IloCplex cplex(model);
        cplex.setOut(env.getNullStream());
        cplex.solve();
        if (cplex.getStatus() == IloAlgorithm::Infeasible) {
            cout << "当前问题不可行" << endl;
            return;
        }
        bool is_feasible = (cplex.getStatus() == IloAlgorithm::Optimal)
                           || (cplex.getStatus() == IloAlgorithm::Feasible);
        while (is_feasible && (cplex.getObjValue() < -1e-4)) {
//            cout << cplex.getObjValue() << endl;
            VectorXd y_opt(2 * m);
            for (int i = 0; i < 2 * m; ++i) {
                y_opt[i] = cplex.getValue(y[i]);
            }
            double violated_value;
            double most_violated_value = -1;
            int most_violated_id = -1;
            for (auto &j:res_cols) {
                violated_value = abs(M.col(j).dot(y_opt));
                if (violated_value > most_violated_value) {
                    most_violated_value = violated_value;
                    most_violated_id = j;
                }
            }
            model.add(cutting_pool[most_violated_id]);
            res_cols.erase(most_violated_id);
            selected_cols.insert(most_violated_id);
//            cout << "非零变量个数：" << selected_cols.size() << endl;
            cplex.solve();
            is_feasible = (cplex.getStatus() == IloAlgorithm::Optimal)
                          || (cplex.getStatus() == IloAlgorithm::Feasible);
        }
        if (is_feasible && (cplex.getObjValue() >= -1e-4)) {
            VectorXd x = VectorXd::Zero(n, 1);
            for (auto &j:selected_cols) {
                x[j] = cplex.getDual(cutting_pool[j]);
            }
            inst.x = x;
        }


    } catch (const IloException &e) {
        cerr << "Exception caught: " << e << endl;
    } catch (...) {
        cerr << "Unknown exception caught!" << endl;
    }

    env.end();
}


void ICP_method(Instance &inst, double alpha = 0.3) {
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
    IloEnv env;
    try {

        IloModel model(env);
        IloFloatVarArray y(env, 2 * m, 0, 1);

        IloExpr obj(env);
        for (int i = 0; i < 2 * m; ++i) {
            obj += c[i] * y[i];
        }
        model.add(IloMinimize(env, obj));

        IloRangeArray cutting_pool(env, n);

        IloExpr expr(env);

        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                expr += -A(i, j) * y[i];
                expr += A(i, j) * y[i + m];
            }
            cutting_pool[j] = expr == 0;
            expr.clear();
        }

        expr.end();

//        model.add(cutting_pool);

        IloCplex cplex(model);
        cplex.setOut(env.getNullStream());
        cplex.solve();
        if (cplex.getStatus() == IloAlgorithm::Infeasible) {
            cout << "当前问题不可行" << endl;
            return;
        }
        bool is_feasible = (cplex.getStatus() == IloAlgorithm::Optimal)
                           || (cplex.getStatus() == IloAlgorithm::Feasible);
        while (is_feasible && (cplex.getObjValue() < -1e-4)) {
//            cout << cplex.getObjValue() << endl;
            VectorXd y_opt(2 * m);
            for (int i = 0; i < 2 * m; ++i) {
                y_opt[i] = cplex.getValue(y[i]);
            }
            double violated_value;
            double most_violated_value = -1;
            int most_violated_id = -1;
            for (auto &j:res_cols) {
                violated_value = abs(M.col(j).dot(y_opt));
                if (violated_value > most_violated_value) {
                    most_violated_value = violated_value;
                    most_violated_id = j;
                }
            }
            model.add(cutting_pool[most_violated_id]);
            res_cols.erase(most_violated_id);
            selected_cols.insert(most_violated_id);
//            cout << "非零变量个数：" << selected_cols.size() << endl;
            cplex.solve();
            is_feasible = (cplex.getStatus() == IloAlgorithm::Optimal)
                          || (cplex.getStatus() == IloAlgorithm::Feasible);
        }


        double k = alpha * selected_cols.size();
        int iter = 0;
        while (k > 1) {
            iter++;
            list <DualValue> J;
            for (auto &j:selected_cols) {
                DualValue dv;
                dv.col = j;
                dv.dual_value = abs(cplex.getDual(cutting_pool[j]));
                J.emplace_back(dv);
            }
            J.sort();
            auto it = J.begin();
            for (int i = 0; i < k; ++i) {
                selected_cols.erase(it->col);
                model.remove(cutting_pool[it->col]);
                res_cols.insert(it->col);
//                cout << it->col << "," << it->dual_value << endl;
                ++it;
            }

            cplex.solve();

            while (is_feasible && (cplex.getObjValue() < -1e-4)) {
//                cout << cplex.getObjValue() << endl;
                VectorXd y_opt(2 * m);
                for (int i = 0; i < 2 * m; ++i) {
                    y_opt[i] = cplex.getValue(y[i]);
                }
                double violated_value;
                double most_violated_value = -1;
                int most_violated_id = -1;
                for (auto &j:res_cols) {
                    violated_value = abs(M.col(j).dot(y_opt));
                    if (violated_value > most_violated_value) {
                        most_violated_value = violated_value;
                        most_violated_id = j;
                    }
                }
                model.add(cutting_pool[most_violated_id]);
                res_cols.erase(most_violated_id);
                selected_cols.insert(most_violated_id);
//            cout << "非零变量个数：" << selected_cols.size() << endl;
                cplex.solve();
                is_feasible = (cplex.getStatus() == IloAlgorithm::Optimal)
                              || (cplex.getStatus() == IloAlgorithm::Feasible);
            }

            k *= 0.8;


        }

        if (is_feasible && (cplex.getObjValue() >= -1e-4)) {
            VectorXd x = VectorXd::Zero(n, 1);
            for (auto &j:selected_cols) {
                x[j] = cplex.getDual(cutting_pool[j]);
            }
            inst.x = x;
        }


    } catch (const IloException &e) {
        cerr << "Exception caught: " << e << endl;
    } catch (...) {
        cerr << "Unknown exception caught!" << endl;
    }

    env.end();
}

#endif //RESOURCE_REALLOCATION_CUTTING_PLANE_H
