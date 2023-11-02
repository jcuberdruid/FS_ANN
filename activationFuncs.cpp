/*
################################################################
# contains the various available activation functions:
sigmoid, tanh and ReLU a
################################################################
*/
#include <cmath>

double sigmoid_act(double input) { // 1 / (1 + e^(-x))
    return 1.0 / (1.0 + exp(-input));
}
double tanh_act(double input) { // (e^(x) - e^(-x)) / (e^(x) + e^(-x))
    return tanh(input);
}
double relu_act(double input) { // max(0, x)
    return (input > 0) ? input : 0;
}
