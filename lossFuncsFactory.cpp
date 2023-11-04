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

vector<double> softmax(const vector<double>& inputs);

double cross_entropy(const vector<double>& outputs, const vector<int>& labels) {
    const vector<double>& outputs_softmax = softmax(outputs);
    double loss = 0.0;
    for (size_t i = 0; i < outputs.size(); ++i) {
        loss -= labels[i] * log(outputs_softmax[i]);
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

// does not conform to factory
vector<double> softmax(const vector<double>& inputs) {
    double max_input = *std::max_element(inputs.begin(), inputs.end());
    vector<double> exp_values(inputs.size());
    double sum_exp_values = 0.0;

    // Compute e^(x - max(x)) for numerical stability
    std::transform(inputs.begin(), inputs.end(), exp_values.begin(),
                   [&max_input](double input) {
                       return exp(input - max_input);
                   });

    sum_exp_values = std::accumulate(exp_values.begin(), exp_values.end(), 0.0);

    // Divide by the sum of all e^x to get probabilities
    std::transform(exp_values.begin(), exp_values.end(), exp_values.begin(),
                   [&sum_exp_values](double value) {
                       return value / sum_exp_values;
                   });

    return exp_values;
}

