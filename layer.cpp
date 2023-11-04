#include <iostream>
#include <vector>
#include <random>
#include "activationFuncsFactory.cpp"

using namespace std;

class Layer; // forward declaration for call backs
using CallBackFunctionType = void (*)(Layer *);

class Optimizer
{
public:
    virtual void backwardPropagate(vector<double> &gradient, double learningRate) = 0;
    Layer *thisLayer;
    void setLayer(Layer *layerPtr)
    {
        this->thisLayer = layerPtr;
    }
};

Optimizer *getOptimizer(string optimizerType, tuple<int, int> shapeWeights);

class Layer
{
public:
    Layer *prev;
    Layer *next;
    vector<vector<double>> weights;
    vector<double> prevLayerGradient;
    vector<double> bias;
    vector<double> lastOutput;
    vector<double> inputVals;
    // vector<double> lastChange; <- part of optimizer now
    Optimizer *optimizer;

    ActivationFunction activationFunction;
    ActivationDerivative activationDerivative;

    vector<CallBackFunctionType> callbacks;

    Layer(const string activationID, tuple<int, int> shapeWeights, string optimizerType, vector<CallBackFunctionType> callbacks)
        : bias(get<0>(shapeWeights), 0.0), callbacks(callbacks), prev(nullptr), next(nullptr)
    {
        initWeights(shapeWeights);
        optimizer = getOptimizer(optimizerType, shapeWeights);
        optimizer->setLayer(this);
        auto [func, deriv] = getActivationFunctions(activationID);
        activationFunction = func;
        activationDerivative = deriv;
    }
    void initWeights(tuple<int, int> shapeWeights)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> distr(-0.5, 0.5);
        int rows = get<0>(shapeWeights);
        int cols = get<1>(shapeWeights);

        weights.resize(rows, vector<double>(cols));
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                weights[i][j] = distr(gen);
            }
        }
    }
    vector<double> forwardPropagate(vector<double> tmp_inputVals)
    { // accept input vals, do matrix mult, add bias, for each in vals activation function, return new output
        inputVals = tmp_inputVals;
        //       cout << "sizeof weights         " << weights[0].size() << endl;
        //       cout << "sizeof input values    " << tmp_inputVals.size() << endl;
        if (weights.size() == 0 || tmp_inputVals.size() != weights[0].size())
        {
            throw std::invalid_argument("Input size does not match weights size.");
        }
        vector<double> activations(weights.size(), 0.0);
        for (size_t i = 0; i < weights.size(); ++i)
        {
            for (size_t j = 0; j < tmp_inputVals.size(); ++j)
            {
                activations[i] += weights[i][j] * tmp_inputVals[j];
            }
        }
        for (size_t i = 0; i < bias.size(); ++i)
        {
            activations[i] += bias[i];
        }
        for (size_t i = 0; i < activations.size(); ++i)
        {
            activations[i] = activationFunction(activations[i]);
        }
        // cout << endl;

        lastOutput = activations;
        return activations;
    }
    void backwardPropagate(vector<double> &gradient, double learningRate)
    {
        optimizer->backwardPropagate(gradient, learningRate);
    }
};

////////////// Optimizer implementations/derived classes

class SGD : public Optimizer
{
public:
    SGD(tuple<int, int> shapeWeights)
    {
    }

    void backwardPropagate(vector<double> &gradient, double learningRate)
    {
        // Compute dOut/dNet (the derivative of the activation function)
        vector<double> dOut_dNet(thisLayer->lastOutput.size());
        for (size_t i = 0; i < thisLayer->lastOutput.size(); ++i)
        {
            dOut_dNet[i] = thisLayer->activationDerivative(thisLayer->lastOutput[i]);
        }

        // Compute dLoss/dNet (gradient with respect to the inputs of the activation function)
        vector<double> dLoss_dNet(gradient.size());
        for (size_t i = 0; i < gradient.size(); ++i)
        {
            dLoss_dNet[i] = gradient[i] * dOut_dNet[i];
        }

        // If there is a previous layer, prepare its gradient
        if (thisLayer->prev)
        {
            thisLayer->prevLayerGradient.resize(thisLayer->weights[0].size(), 0.0);
            for (size_t i = 0; i < thisLayer->weights.size(); ++i)
            {
                for (size_t j = 0; j < thisLayer->weights[0].size(); ++j)
                {
                    thisLayer->prevLayerGradient[j] += dLoss_dNet[i] * thisLayer->weights[i][j];
                }
            }
        }

        // Compute gradients with respect to thisLayer->weights and thisLayer->biases
        for (size_t i = 0; i < thisLayer->weights.size(); ++i)
        {
            for (size_t j = 0; j < thisLayer->weights[i].size(); ++j)
            {
                // Update the weight by subtracting the learning rate times the gradient
                thisLayer->weights[i][j] -= learningRate * dLoss_dNet[i] * (thisLayer->prev ? thisLayer->prev->lastOutput[j] : thisLayer->inputVals[j]); // thisLayer->inputVals should be the input to the current layer
            }
            // Update the thisLayer->bias by subtracting the learning rate times the gradient
            thisLayer->bias[i] -= learningRate * dLoss_dNet[i];
        }
    }
};

class SGDwMomentum : public Optimizer
{
public:
    double momentum = 0.3;
    vector<double> lastChange;

    SGDwMomentum(tuple<int, int> shapeWeights) : lastChange(get<0>(shapeWeights), 0.0)
    {
    }

    void backwardPropagate(vector<double> &gradient, double learningRate)
    {
        vector<double> dOut_dNet(thisLayer->lastOutput.size());
        for (size_t i = 0; i < thisLayer->lastOutput.size(); ++i)
        {
            dOut_dNet[i] = thisLayer->activationDerivative(thisLayer->lastOutput[i]);
        }

        vector<double> dLoss_dNet(gradient.size());
        for (size_t i = 0; i < gradient.size(); ++i)
        {
            dLoss_dNet[i] = gradient[i] * dOut_dNet[i];
        }

        if (thisLayer->prev)
        {
            thisLayer->prevLayerGradient.resize(thisLayer->weights[0].size(), 0.0);
            for (size_t i = 0; i < thisLayer->weights.size(); ++i)
            {
                for (size_t j = 0; j < thisLayer->weights[0].size(); ++j)
                {
                    thisLayer->prevLayerGradient[j] += dLoss_dNet[i] * thisLayer->weights[i][j];
                }
            }
        }

        for (size_t i = 0; i < thisLayer->weights.size(); ++i)
        {
            for (size_t j = 0; j < thisLayer->weights[i].size(); ++j)
            {
                auto newChange = learningRate * dLoss_dNet[i] * (thisLayer->prev ? thisLayer->prev->lastOutput[j] : thisLayer->inputVals[j]) + momentum * lastChange[i];
                thisLayer->weights[i][j] -= newChange;
                lastChange[i] = newChange;
            }
            thisLayer->bias[i] -= learningRate * dLoss_dNet[i];
        }
    }
};

class AdaGrad : public Optimizer
{
private:
    vector<vector<double>> gradientHistory;

public:
    AdaGrad(tuple<int, int> shapeWeights)
    {
        int rows = get<0>(shapeWeights);
        int cols = get<1>(shapeWeights);
        gradientHistory.resize(rows, vector<double>(cols, 0.0));
    }

    void backwardPropagate(vector<double> &gradient, double learningRate)
    {
        vector<double> dOut_dNet(thisLayer->lastOutput.size());
        for (size_t i = 0; i < thisLayer->lastOutput.size(); ++i)
        {
            dOut_dNet[i] = thisLayer->activationDerivative(thisLayer->lastOutput[i]);
        }

        vector<double> dLoss_dNet(gradient.size());
        for (size_t i = 0; i < gradient.size(); ++i)
        {
            dLoss_dNet[i] = gradient[i] * dOut_dNet[i];
        }

        if (thisLayer->prev)
        {
            thisLayer->prevLayerGradient.resize(thisLayer->weights[0].size(), 0.0);
            for (size_t i = 0; i < thisLayer->weights.size(); ++i)
            {
                for (size_t j = 0; j < thisLayer->weights[0].size(); ++j)
                {
                    thisLayer->prevLayerGradient[j] += dLoss_dNet[i] * thisLayer->weights[i][j];
                }
            }
        }

        for (size_t i = 0; i < thisLayer->weights.size(); ++i)
        {
            for (size_t j = 0; j < thisLayer->weights[i].size(); ++j)
            {
                double gradient = dLoss_dNet[i] * (thisLayer->prev ? thisLayer->prev->lastOutput[j] : thisLayer->inputVals[j]);
                gradientHistory[i][j] += std::pow(gradient, 2);

                double adjustedLearningRate = learningRate / (sqrt(gradientHistory[i][j]) + 1e-7);
                thisLayer->weights[i][j] -= adjustedLearningRate * gradient;
            }
            double gradient = dLoss_dNet[i];
            gradientHistory[i][thisLayer->bias.size()] += std::pow(gradient, 2);
            double adjustedLearningRateBias = learningRate / (sqrt(gradientHistory[i][thisLayer->bias.size()]) + 1e-7);
            thisLayer->bias[i] -= adjustedLearningRateBias * gradient;
        }
    }
};

class RMSProp : public Optimizer
{
private:
    double decayRate = 0.9;
    double epsilon = 1e-8;
    vector<vector<double>> squaredGradientAverage;

public:
    RMSProp(tuple<int, int> shapeWeights)
    {
        int rows = get<0>(shapeWeights);
        int cols = get<1>(shapeWeights);
        squaredGradientAverage.resize(rows, vector<double>(cols, 0.0));
    }

    void backwardPropagate(vector<double> &gradient, double learningRate)
    {
        vector<double> dOut_dNet(thisLayer->lastOutput.size());
        for (size_t i = 0; i < thisLayer->lastOutput.size(); ++i)
        {
            dOut_dNet[i] = thisLayer->activationDerivative(thisLayer->lastOutput[i]);
        }

        vector<double> dLoss_dNet(gradient.size());
        for (size_t i = 0; i < gradient.size(); ++i)
        {
            dLoss_dNet[i] = gradient[i] * dOut_dNet[i];
        }

        if (thisLayer->prev)
        {
            thisLayer->prevLayerGradient.resize(thisLayer->weights[0].size(), 0.0);
            for (size_t i = 0; i < thisLayer->weights.size(); ++i)
            {
                for (size_t j = 0; j < thisLayer->weights[0].size(); ++j)
                {
                    thisLayer->prevLayerGradient[j] += dLoss_dNet[i] * thisLayer->weights[i][j];
                }
            }
        }

        for (size_t i = 0; i < thisLayer->weights.size(); ++i)
        {
            for (size_t j = 0; j < thisLayer->weights[i].size(); ++j)
            {
                double gradient = dLoss_dNet[i] * (thisLayer->prev ? thisLayer->prev->lastOutput[j] : thisLayer->inputVals[j]);
                squaredGradientAverage[i][j] = decayRate * squaredGradientAverage[i][j] + (1 - decayRate) * gradient * gradient;
                double adjustedLearningRate = learningRate / (sqrt(squaredGradientAverage[i][j]) + epsilon);
                thisLayer->weights[i][j] -= adjustedLearningRate * gradient;
            }
            double biasGradient = dLoss_dNet[i];
            squaredGradientAverage[i][thisLayer->bias.size()] = decayRate * squaredGradientAverage[i][thisLayer->bias.size()] + (1 - decayRate) * biasGradient * biasGradient;
            double adjustedLearningRateBias = learningRate / (sqrt(squaredGradientAverage[i][thisLayer->bias.size()]) + epsilon);
            thisLayer->bias[i] -= adjustedLearningRateBias * biasGradient;
        }
    }
};

class Adam : public Optimizer {
private:
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;
    vector<vector<double>> m;
    vector<vector<double>> v;

public:
    Adam(tuple<int, int> shapeWeights) {
        int rows = get<0>(shapeWeights);
        int cols = get<1>(shapeWeights);
        m.resize(rows, vector<double>(cols, 0.0));
        v.resize(rows, vector<double>(cols, 0.0));
    }

    void backwardPropagate(vector<double> &gradient, double learningRate) {
        t++;
        vector<double> dOut_dNet(thisLayer->lastOutput.size());
        for (size_t i = 0; i < thisLayer->lastOutput.size(); ++i) {
            dOut_dNet[i] = thisLayer->activationDerivative(thisLayer->lastOutput[i]);
        }

        vector<double> dLoss_dNet(gradient.size());
        for (size_t i = 0; i < gradient.size(); ++i) {
            dLoss_dNet[i] = gradient[i] * dOut_dNet[i];
        }

        if (thisLayer->prev) {
            thisLayer->prevLayerGradient.resize(thisLayer->weights[0].size(), 0.0);
            for (size_t i = 0; i < thisLayer->weights.size(); ++i) {
                for (size_t j = 0; j < thisLayer->weights[0].size(); ++j) {
                    thisLayer->prevLayerGradient[j] += dLoss_dNet[i] * thisLayer->weights[i][j];
                }
            }
        }

        for (size_t i = 0; i < thisLayer->weights.size(); ++i) {
            for (size_t j = 0; j < thisLayer->weights[i].size(); ++j) {
                double grad = dLoss_dNet[i] * (thisLayer->prev ? thisLayer->prev->lastOutput[j] : thisLayer->inputVals[j]);
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad;
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad * grad;
                double m_hat = m[i][j] / (1 - pow(beta1, t));
                double v_hat = v[i][j] / (1 - pow(beta2, t));
                thisLayer->weights[i][j] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
            }
            double biasGrad = dLoss_dNet[i];
            m[i][thisLayer->bias.size()] = beta1 * m[i][thisLayer->bias.size()] + (1 - beta1) * biasGrad;
            v[i][thisLayer->bias.size()] = beta2 * v[i][thisLayer->bias.size()] + (1 - beta2) * biasGrad * biasGrad;
            double m_hat_bias = m[i][thisLayer->bias.size()] / (1 - pow(beta1, t));
            double v_hat_bias = v[i][thisLayer->bias.size()] / (1 - pow(beta2, t));
            thisLayer->bias[i] -= learningRate * m_hat_bias / (sqrt(v_hat_bias) + epsilon);
        }
    }
};


Optimizer *getOptimizer(string optimizerType, tuple<int, int> shapeWeights)
{
    if (optimizerType == "sgd")
    {
        return new SGD(shapeWeights);
    }
    else if (optimizerType == "sgd_momentum")
    {
        return new SGDwMomentum(shapeWeights);
    }
    else if (optimizerType == "adagrad") {
        return new AdaGrad(shapeWeights);
    }
    else if (optimizerType == "rmsprop") {
        return new RMSProp(shapeWeights);
    }
    else if (optimizerType == "adam") {
        return new Adam(shapeWeights);
    }
    return new SGD(shapeWeights);
}
