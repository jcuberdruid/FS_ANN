#include<iostream>
#include<vector>
#include<random>
#include"activationFuncsFactory.cpp"

using namespace std;
//using ActivFunctionType = double(*)(double); //for function pointers //unused currently

class Layer; //forward declaration for call backs
using CallBackFunctionType = void(*)(Layer*); 

class Layer {
    public: 
        Layer* prev;
        Layer* next;
        vector<vector<double>> weights;
        vector<double> prevLayerGradient;
        vector<double> bias;
        vector<double> lastOutput;
        vector<double> inputVals;
        ActivationFunction activationFunction;
        ActivationDerivative activationDerivative;
        
        vector<CallBackFunctionType> callbacks;
    
    Layer(const string activationID, tuple<int, int> shapeWeights, vector<CallBackFunctionType> callbacks) 
       :  bias(get<0>(shapeWeights), 0.0), callbacks(callbacks), prev(nullptr), next(nullptr) 
    {
                initWeights(shapeWeights);
                auto [func, deriv] = getActivationFunctions(activationID);
                activationFunction = func;
                activationDerivative = deriv;
    }
    void initWeights(tuple<int, int> shapeWeights) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> distr(-0.5, 0.5);
        int rows = get<0>(shapeWeights);
        int cols = get<1>(shapeWeights);

        weights.resize(rows, vector<double>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                weights[i][j] = distr(gen);
            }
        }
    }
    vector<double> forwardPropagate(vector<double> tmp_inputVals) { //accept input vals, do matrix mult, add bias, for each in vals activation function, return new output 
        inputVals = tmp_inputVals;
 //       cout << "sizeof weights         " << weights[0].size() << endl;
 //       cout << "sizeof input values    " << tmp_inputVals.size() << endl;
        if (weights.size() == 0 || tmp_inputVals.size() != weights[0].size()) {
            throw std::invalid_argument("Input size does not match weights size.");
        }
        vector<double> activations(weights.size(), 0.0); 
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < tmp_inputVals.size(); ++j) {
                activations[i] += weights[i][j] * tmp_inputVals[j];
            }
        }
        for (size_t i = 0; i < bias.size(); ++i) {
            activations[i] += bias[i];
        }
        for (size_t i = 0; i < activations.size(); ++i) {
            activations[i] = activationFunction(activations[i]);
        }
       // cout << endl;
        
        lastOutput = activations;
        return activations;
    }
    void backwardPropagate(vector<double>& gradient, double learningRate) {
    // Compute dOut/dNet (the derivative of the activation function)
    vector<double> dOut_dNet(lastOutput.size());
    for (size_t i = 0; i < lastOutput.size(); ++i) {
        dOut_dNet[i] = activationDerivative(lastOutput[i]);
    }

    // Compute dLoss/dNet (gradient with respect to the inputs of the activation function)
    vector<double> dLoss_dNet(gradient.size());
    for (size_t i = 0; i < gradient.size(); ++i) {
        dLoss_dNet[i] = gradient[i] * dOut_dNet[i];
    }

    // If there is a previous layer, prepare its gradient
    if (prev) {
        prevLayerGradient.resize(weights[0].size(), 0.0);
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[0].size(); ++j) {
                prevLayerGradient[j] += dLoss_dNet[i] * weights[i][j];
            }
        }
    }

    // Compute gradients with respect to weights and biases
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            // Update the weight by subtracting the learning rate times the gradient
            weights[i][j] -= learningRate * dLoss_dNet[i] * (prev ? prev->lastOutput[j] : inputVals[j]); // inputVals should be the input to the current layer
        }
        // Update the bias by subtracting the learning rate times the gradient
        bias[i] -= learningRate * dLoss_dNet[i];
    }

}

};

/*
################################################################
# Example function pointers: activation and call back
################################################################
*/
double activationFunctionTest(double x) {
    cout << "test function" << endl;
    double test = 566.7;
    return test;
}

void exampleCallback(Layer* layer) {
    if (layer != nullptr) {
        cout << "Callback called on Layer instance. First weight is: ";
       if (!layer->weights.empty()) {
           cout << layer->weights[0].front() << endl;
        }
    }
}


/*
################################################################
# testing main function  
################################################################
int main() {
    cout << "main" << endl;
  
    vector<CallBackFunctionType> myCallbacks = {exampleCallback}; 
    Layer bla(activationFunctionTest, 5, make_tuple(5, 7), myCallbacks);

    for (auto& callback : bla.callbacks) {
        callback(&bla);
    }
   
    vector<double> testVal(7, 14.0);

    vector<double> testoutput = bla.forwardPropagate(testVal);

    for(auto it: testoutput) {
        cout << it << endl;
    }

    return 0; 
}

*/
