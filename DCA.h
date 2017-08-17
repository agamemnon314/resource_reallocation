//
// Created by agamemnon on 17-8-17.
//

#ifndef RESOURCE_REALLOCATION_DCA_H
#define RESOURCE_REALLOCATION_DCA_H

#include <ilcplex/ilocplex.h>
#include <iostream>
#include <string>
#include "instance.h"


ILOSTLBEGIN

double sgn(const double x) {
    double s = 0;
    if (x > 0) {
        s = 1;
    }
    if (x == 0) {
        s = 0;
    }
    if (x < 0) {
        s = -1;
    }
    return s;
}


void d_PiE(VectorXd &x, VectorXd &y, const double lambda) {
    for (int i = 0; i < x.rows(); ++i) {
        y[i] = sgn(x[i]) * (lambda * abs(x[i]) - exp(-lambda * abs(x[i])));
    }

}

void d_Cap(VectorXd &x, VectorXd &y, double lambda) {
    for (int i = 0; i < x.rows(); ++i) {
        if (abs(x[i]) <= 1 / lambda) {
            y[i] = 0;
        } else {
            y[i] = sgn(x[i]) * lambda;
        }
    }

}

void d_SCAD(VectorXd &x, VectorXd &y, const double lambda, const double alpha) {
    for (int j = 0; j < x.rows(); ++j) {
        if (abs(x[j]) <= lambda) {
            y[j] = 0;
        } else {
            if (abs(x[j]) < alpha * lambda) {
                y[j] = sgn(x[j]) * (abs(x[j]) - lambda) / (alpha - 1);
            } else {
                y[j] = sgn(x[j]) * lambda;
            }
        }
    }

}

void d_PiL(VectorXd &x, VectorXd &y, const double a, const double b) {
    for (int i = 0; i < x.rows(); ++i) {
        if (abs(x[i]) < b) {
            y[i] = 0;
        } else {
            y[i] = sgn(x[i]) / (b - a);
        }
    }

}


void DCA_Cap(Instance &inst, double lamba = 0.05, double tol = 1e-4) {
    MatrixXd &A = inst.A;
    VectorXd &l = inst.l;
    VectorXd &u = inst.u;
    const int m = A.rows();
    const int n = A.cols();

    VectorXd x_cur = VectorXd::Zero(n, 1);
    VectorXd x_next = VectorXd::Zero(n, 1);
    VectorXd y = VectorXd::Zero(n, 1);

    IloEnv env;
    try {

        IloModel model(env);
        IloNumVarArray x(env, n, -IloInfinity, IloInfinity, ILOFLOAT);
        IloNumVarArray t(env, n, 0, IloInfinity, ILOFLOAT);


        d_Cap(x_cur, y, lamba);


        IloExpr obj_expr(env);
        IloObjective obj(env);
        for (int j = 0; j < n; ++j) {
            obj_expr += lamba * t[j];
            obj_expr += -y[j] * x[j];
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
            cout << step_size << endl;
            for (int j = 0; j < n; ++j) {
                x_next[j] = cplex.getValue(x[j]);
            }
            step_size = (x_next - x_cur).norm() / (x_cur.norm() + 1);
            x_cur = x_next;
            d_Cap(x_cur, y, lamba);


            model.remove(obj);
            for (int j = 0; j < n; ++j) {
                obj_expr += lamba * t[j];
                obj_expr += -y[j] * x[j];
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

void DCA_SCAD(Instance &inst, double lamba = 0.05, double alpha = 40, double tol = 1e-4) {
    MatrixXd &A = inst.A;
    VectorXd &l = inst.l;
    VectorXd &u = inst.u;
    const int m = A.rows();
    const int n = A.cols();

    VectorXd x_cur = VectorXd::Zero(n, 1);
    VectorXd x_next = VectorXd::Zero(n, 1);
    VectorXd y = VectorXd::Zero(n, 1);

    IloEnv env;
    try {

        IloModel model(env);
        IloNumVarArray x(env, n, -IloInfinity, IloInfinity, ILOFLOAT);
        IloNumVarArray t(env, n, 0, IloInfinity, ILOFLOAT);


        d_SCAD(x_cur, y, lamba, alpha);


        IloExpr obj_expr(env);
        IloObjective obj(env);
        for (int j = 0; j < n; ++j) {
            obj_expr += lamba * t[j];
            obj_expr += -y[j] * x[j];
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
            cout << step_size << endl;
            for (int j = 0; j < n; ++j) {
                x_next[j] = cplex.getValue(x[j]);
            }
            step_size = (x_next - x_cur).norm() / (x_cur.norm() + 1);
            x_cur = x_next;
            d_SCAD(x_cur, y, lamba, alpha);


            model.remove(obj);
            for (int j = 0; j < n; ++j) {
                obj_expr += lamba * t[j];
                obj_expr += -y[j] * x[j];
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

void DCA_PiL(Instance &inst, double a = 0.001, double b = 0.101, double tol = 1e-4) {
    MatrixXd &A = inst.A;
    VectorXd &l = inst.l;
    VectorXd &u = inst.u;
    const int m = A.rows();
    const int n = A.cols();

    VectorXd x_cur = VectorXd::Zero(n, 1);
    VectorXd x_next = VectorXd::Zero(n, 1);
    VectorXd y = VectorXd::Zero(n, 1);

    IloEnv env;
    try {

        IloModel model(env);
        IloNumVarArray x(env, n, -IloInfinity, IloInfinity, ILOFLOAT);
        IloNumVarArray t(env, n, 0, IloInfinity, ILOFLOAT);


        d_PiL(x_cur, y, a, b);


        IloExpr obj_expr(env);
        IloObjective obj(env);
        for (int j = 0; j < n; ++j) {
            obj_expr += t[j] / (b - a);
            obj_expr += -y[j] * x[j];
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
            cout << step_size << endl;
            for (int j = 0; j < n; ++j) {
                x_next[j] = cplex.getValue(x[j]);
            }
            step_size = (x_next - x_cur).norm() / (x_cur.norm() + 1);
            x_cur = x_next;
            d_PiL(x_cur, y, a, b);


            model.remove(obj);
            for (int j = 0; j < n; ++j) {
                obj_expr += t[j] / (b - a);
                obj_expr += -y[j] * x[j];
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

#endif //RESOURCE_REALLOCATION_DCA_H
