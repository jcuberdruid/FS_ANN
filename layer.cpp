#include<iostream>
#include<vector>
#include<random>

using namespace std;
using ActivFunctionType = double(*)(double); //for function pointers 

class Layer; //forward declaration for call backs
using CallBackFunctionType = void(*)(Layer*); 

class Layer {
    public: 
        Layer* prev;
        Layer* next;
        vector<vector<double>> weights;
        vector<double> bias;
        ActivFunctionType activationFunction;
        vector<CallBackFunctionType> callbacks;
    
    Layer(ActivFunctionType activationFunction, size_t numBias, tuple<int, int> shapeWeights, vector<CallBackFunctionType> callbacks) 
       :  activationFunction(activationFunction), bias(numBias, 0.0), callbacks(callbacks), prev(nullptr), next(nullptr) 
    {
                initWeights(shapeWeights);
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
    vector<double> forwardPropagate(vector<double> inputVals) { //accept input vals, do matrix mult, add bias, for each in vals activation function, return new output 
        cout << weights[0].size() << endl;;
        cout << inputVals.size() << endl;;
        if (weights.size() == 0 || inputVals.size() != weights[0].size()) {
            throw std::invalid_argument("Input size does not match weights size.");
        }
        vector<double> activations(weights.size(), 0.0); 
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < inputVals.size(); ++j) {
                activations[i] += weights[i][j] * inputVals[j];
            }
        }
        for (size_t i = 0; i < bias.size(); ++i) {
            activations[i] += bias[i];
        }
        for (size_t i = 0; i < activations.size(); ++i) {
            activations[i] = activationFunction(activations[i]);
        }
        for(auto it :  activations) {
            cout << it;
        } 
        cout << endl;
        return activations;
    }
    void callAllCallBacks() {
        //go through callback list and call functions
        cout << "unfinished" << endl;
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
