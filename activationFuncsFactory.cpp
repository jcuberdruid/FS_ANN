#include <functional>
#include <string>
#include <unordered_map>
#include <utility>

using ActivationFunction = std::function<double(double)>;
using ActivationDerivative = std::function<double(double)>;
using ActivationPair = std::pair<ActivationFunction, ActivationDerivative>;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu(double x) {
    return std::max(0.0, x);
}
double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}


ActivationPair getActivationFunctions(const std::string& name) {
    static std::unordered_map<std::string, ActivationPair> activation_map = {
        {"sigmoid", {sigmoid, sigmoid_derivative}},
        {"relu", {relu, relu_derivative}},
        // ... Add other mappings as needed ...
    };

    auto it = activation_map.find(name);
    if (it != activation_map.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Unknown activation function: " + name);
    }
}

// auto [actFunc, actDeriv] = getActivationFunctions("sigmoid");

