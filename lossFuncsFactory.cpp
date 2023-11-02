#include <vector>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <stdexcept>

using LossFunction = std::function<double(const std::vector<double>&, const std::vector<int>&)>;
using LossDerivative = std::function<std::vector<double>(const std::vector<double>&, const std::vector<int>&)>;
using LossPair = std::pair<LossFunction, LossDerivative>;

double cross_entropy(const vector<double>& outputs, const vector<int>& labels) {
    double loss = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        loss -= labels[i] * log(outputs[i]);
    }
    return loss;
}

vector<double> softmax_cross_entropy_derivative(const vector<double>& outputs, const vector<int>& labels) {
    vector<double> derivatives(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        derivatives[i] = outputs[i] - labels[i];
    }
    return derivatives;
}

LossPair getLossFunctions(const string& name) {
    static unordered_map<string, LossPair> loss_map = {
        {"cross_entropy", {cross_entropy, softmax_cross_entropy_derivative}},
    };

    auto it = loss_map.find(name);
    if (it != loss_map.end()) {
        return it->second;
    } else {
        throw invalid_argument("Unknown loss function: " + name);
    }
}

// auto [lossFunc, lossDeriv] = getLossFunctions("cross_entropy");

