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


/*
################################################################
# derivatives of activation functions (yeah should probably just make a clases and an interface for this)
################################################################
*/
double sigmoid_derivative(double input) {
    // Sigmoid derivative: σ(x)(1 - σ(x))
    double sigmoidValue = sigmoid_act(input);
    return sigmoidValue * (1.0 - sigmoidValue);
}

double tanh_derivative(double input) {
    // Tanh derivative: 1 - tanh^2(x)
    double tanhValue = tanh_act(input);
    return 1.0 - tanhValue * tanhValue;
}

double relu_derivative(double input) {
    // ReLU derivative: 1 if input > 0 else 0
    return (input > 0) ? 1.0 : 0.0;
}

