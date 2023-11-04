#include "layer.cpp"
#include "lossFuncsFactory.cpp"
#include "utils.cpp"
#include "optimizer.cpp"
#include <iomanip>
#include <fstream>
#include <random>
#include <algorithm>
using namespace std;
using FunctionType = double (*)(double); // for function pointers
using CallBackFunctionType = void (*)(Layer *);

class Model
{
public:
    bool train = true;
    double sumLoss = 0.0;
    int totalPredictions = 0;
    int correctPredictions = 0;
    float lastAccuracy;
    LossFunction lossFunction;
    LossDerivative lossDerivative;
    Optimizer optimizer;
    double learningRate;
    // int batchSize; <- mayber not
    // optimizer <- interface
    // data <- either vector or a class of amorphous shaping
    vector<int> labels;
    Layer *topographyHead = NULL;
    vector<CallBackFunctionType> callBacks;

    Model(const string lossFunctionID, Optimizer optimizer, double learningRate) : optimizer(optimizer), learningRate(learningRate)
    {

        auto [lossFunc, lossDeriv] = getLossFunctions(lossFunctionID);
        lossFunction = lossFunc;
        lossDerivative = lossDeriv;
    }
    void addLayer(string activationID, tuple<int, int> shapeWeights, vector<CallBackFunctionType> callbacks)
    {
        Layer *newLayer = new Layer(activationID, shapeWeights, callbacks);

        if (topographyHead == NULL)
        {
            topographyHead = newLayer;
        }
        else
        {
            Layer *thisLayer = topographyHead;
            while (thisLayer->next != NULL)
            {
                thisLayer = thisLayer->next;
            }
            thisLayer->next = newLayer;
            newLayer->prev = thisLayer;
        }
    }
    ~Model()
    { // destructor to remove layers when model instance is destroy (on the off chance we run multiple models, but at least better than keras)
        Layer *current = topographyHead;
        while (current != NULL)
        {
            Layer *nextLayer = current->next;
            delete current;
            current = nextLayer;
        }
    }
    float getLastAccuracy()
    {
        return lastAccuracy;
    }
    void resetAccuracy()
    {
        totalPredictions = 0;
        correctPredictions = 0;
    }
    /*
        teach(
                data
                labels
                epochs
                callBacks
                validationSplit
                useOptimizedLibraries=False <- (use libraries for matrix multiplication)
                    )

        data -> vector<vector<float>>

        loop epochs:
            for x in


    */
    // Function to shuffle two vectors of vector<int> in unison
    void shuffleInUnison(std::vector<std::vector<int>> &data, std::vector<std::vector<int>> &labels)
    {
        std::random_device rd;
        std::mt19937 generator(rd());

        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::shuffle(indices.begin(), indices.end(), generator);

        std::vector<std::vector<int>> new_data(data.size(), std::vector<int>());
        std::vector<std::vector<int>> new_labels(labels.size(), std::vector<int>());

        for (size_t i = 0; i < indices.size(); i++)
        {
            new_data[i] = data[indices[i]];
            new_labels[i] = labels[indices[i]];
        }

        data.swap(new_data);
        labels.swap(new_labels);
    }
    void normalizeVector(vector<double> &v)
    {
        // Find the minimum and maximum values in the vector
        auto min_max = std::minmax_element(v.begin(), v.end());
        double min_val = *min_max.first;
        double max_val = *min_max.second;

        // Avoid division by zero
        if (min_val == max_val)
        {
            std::fill(v.begin(), v.end(), 0.0f); // or handle the case as needed
            return;
        }

        // Apply normalization: (x - min) / (max - min)
        for (auto &element : v)
        {
            element = (element - min_val) / (max_val - min_val);
        }
    }
    void teach(vector<vector<int>> label_vec, vector<vector<int>> images, int epochs)
    {
        train = true;

        for (int j = 1; j < (epochs + 1); j++)
        {
            shuffleInUnison(images, label_vec);
            const int barWidth = 70;
            for (size_t i = 0; i < label_vec.size(); ++i)
            {
                vector<double> tmp = cast_vector_to_double(images[i]);
                // normalizeVector(tmp);
                epoch(tmp, label_vec[i]);

                double progress = static_cast<double>(i) / label_vec.size();

                cout << "[";
                int pos = static_cast<int>(barWidth * progress);
                for (int j = 0; j < barWidth; ++j)
                {
                    if (j < pos)
                        cout << "=";
                    else if (j == pos)
                        cout << ">";
                    else
                        cout << " ";
                }
                cout << "epoch " << j << "/" << epochs << "] " << int(progress * 100.0) << " % , accuracy: " << getLastAccuracy() << " %\r";
                cout.flush();
            }
            cout << endl;
            ofstream myfile;
            myfile.open("training_accuracyLog_tanh_RMS.txt", ios::app);
            myfile << j << "," << float(correctPredictions) / float(totalPredictions) << "," << sumLoss / double(totalPredictions) << "\n";
            sumLoss = 0.0;
            myfile.close();
            resetAccuracy();
        }
        train = false;
    }
    void epoch(vector<double> input_data, vector<int> input_labels)
    {
        vector<double> modelOutput = forwardPropagate(input_data);
        // cout << "######################" << endl;
        sumLoss += lossFunction(modelOutput, input_labels);
        vector<double> softmaxOutput = softmax(modelOutput);
        // cout << "######################" << endl;
        /*
        for(auto it: input_data) {
            cout << " " << it;
        }
        cout << endl;
        for(auto it: input_labels) {
            cout << it << "           ";
        }
        cout << endl;
        for(auto it: softmaxOutput) {
            cout << " " << it;
        }
        cout << endl;
        */
        auto max_it = std::max_element(softmaxOutput.begin(), softmaxOutput.end()); // XXX max function can't handle equal inputs
        int max_index = std::distance(softmaxOutput.begin(), max_it);
        if (input_labels[max_index] == 1)
        {
            correctPredictions += 1;
            totalPredictions += 1;
        }
        else
        {
            totalPredictions += 1;
        }
        // cout << "accuracy: " << float(correctPredictions)/float(totalPredictions) << endl;

        lastAccuracy = float(correctPredictions) / float(totalPredictions);

        // cout << "######################" << endl;
        if (train == true)
        {
            backPropagate(modelOutput, input_labels);
        }
    }
    void predict(vector<vector<int>> label_vec, vector<vector<int>> images)
    {
        int epochs = 1;
        train = false;
        for (int j = 1; j < (epochs + 1); j++)
        {
            const int barWidth = 70;
            for (size_t i = 0; i < label_vec.size(); ++i)
            {
                vector<double> tmp = cast_vector_to_double(images[i]);
                epoch(tmp, label_vec[i]);

                double progress = static_cast<double>(i) / label_vec.size();

                cout << "[";
                int pos = static_cast<int>(barWidth * progress);
                for (int j = 0; j < barWidth; ++j)
                {
                    if (j < pos)
                        cout << "=";
                    else if (j == pos)
                        cout << ">";
                    else
                        cout << " ";
                }
                cout << "testing "
                     << "/" << epochs << "] " << int(progress * 100.0) << " % , accuracy: " << getLastAccuracy() << " %\r";
                cout.flush();
            }
            cout << endl;
            ofstream myfile;
            myfile.open("testing_accuracyLog_tanh_RMS.txt", ios::app);
            myfile << j << "," << float(correctPredictions) / float(totalPredictions) << "," << sumLoss / double(totalPredictions) << "\n";
            sumLoss = 0.0;
            myfile.close();
            resetAccuracy();
        }
        train = true;
    }
    void infoLayers()
    {
        cout << "Layer (type)\t\tWeight Matrix Shape\t\tParam #" << endl;
        cout << "=======================================================================" << endl;

        Layer *current = topographyHead;
        int totalParams = 0;
        int numLayers = 0;

        while (current != NULL)
        {
            auto shapeWeights = make_tuple(current->weights.size(), current->weights.empty() ? 0 : current->weights[0].size());
            int layerParams = get<0>(shapeWeights) * get<1>(shapeWeights) + get<0>(shapeWeights); // Weights + Bias
            cout << setw(10) << numLayers << " (Layer)\t\t"
                 << "[" << get<0>(shapeWeights) << "," << get<1>(shapeWeights) << "]\t\t"
                 << layerParams << endl;

            totalParams += layerParams;
            numLayers++;
            current = current->next;
        }
        cout << "=======================================================================" << endl;
        cout << "Total params: " << totalParams << endl;
        cout << "Trainable params: " << totalParams << endl;
        cout << "Non-trainable params: 0" << endl;
        cout << "Total layers: " << numLayers << endl;
    }

private:
    vector<double> forwardPropagate(vector<double> input_data)
    {
        Layer *current = topographyHead;

        while (current != NULL)
        {
            input_data = current->forwardPropagate(input_data);
            current = current->next;
        }
        return input_data;
    }

    void backPropagate(vector<double> &output, vector<int> &trueLabels)
    {
        vector<double> gradient = lossDerivative(output, trueLabels);

        std::vector<Layer *> layersToUpdate;
        Layer *current = topographyHead;
        while (current)
        {
            layersToUpdate.push_back(current);
            current = current->next;
        }
        std::reverse(layersToUpdate.begin(), layersToUpdate.end());

        for (Layer *layer : layersToUpdate)
        {
            layer->backwardPropagate(gradient, learningRate);
            gradient = layer->prevLayerGradient;
        }
    }
};
