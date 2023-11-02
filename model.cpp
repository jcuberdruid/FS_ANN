/*
################################
# Model Class: handles layers, sort of a controller, but also user facing 
################################

model class 
    vars
    // loss function 
    // learning rate 
    // batchsize <- haven't implemented batching though so maybe not (or mayber later)
    // optimizer 
    // data 
    // labels 
    // model topography (holds layers)
   
    functions external:
    model.
        .init(loss, optimizer, learningRate) <- maybe learning rate should be optimizer param but oh well 
        .addLayer(layer params: ) <- maybe eventually have different layers
            ActivFunctionType activationFunction, size_t numBias, tuple<int, int> shapeWeights, vector<CallBackFunctionType> callbacks
        .teach(
                data 
                labels
                epochs
                callBacks
                validationSplit
                useOptimizedLibraries=False <- (use libraries for matrix multiplication) 
                    ) 
        .infoLayers <- same thing as model.summary() in keras  
                
    functions internal 
        forwardProgate 
        backpropagate
        calcLoss
                
*/
#include"layer.cpp"
#include"lossFuncsFactory.cpp"
#include"optimizer.cpp"
#include <iomanip>
#include<fstream>
using namespace std;
using FunctionType = double(*)(double); //for function pointers 
using CallBackFunctionType = void(*)(Layer*);

class Model {
 public: 
    int totalPredictions = 0;
    int correctPredictions = 0;
    LossFunction lossFunction;        
    LossDerivative lossDerivative;
    Optimizer optimizer;
    double learningRate;
    //int batchSize; <- mayber not 
    //optimizer <- interface 
    //data <- either vector or a class of amorphous shaping     
    vector<int> labels;
    Layer* topographyHead = NULL; 
    vector<CallBackFunctionType> callBacks; 
    
    Model(const string lossFunctionID, Optimizer optimizer, double learningRate) :  optimizer(optimizer), learningRate(learningRate){

        auto [lossFunc, lossDeriv] = getLossFunctions(lossFunctionID);
        lossFunction = lossFunc;
        lossDerivative = lossDeriv;
        }
    void addLayer(string activationID, tuple<int, int> shapeWeights, vector<CallBackFunctionType> callbacks) {
        Layer* newLayer = new Layer(activationID, shapeWeights, callbacks);
        
        if (topographyHead == NULL) {
            topographyHead = newLayer;
        } else {
            Layer* thisLayer = topographyHead;
            while (thisLayer->next != NULL) {
                thisLayer = thisLayer->next;
            }
            thisLayer->next = newLayer;
            newLayer->prev = thisLayer; 
        }
    }
    ~Model() { // destructor to remove layers when model instance is destroy (on the off chance we run multiple models, but at least better than keras)
        Layer* current = topographyHead;
        while (current != NULL) {
            Layer* nextLayer = current->next;
            delete current;
            current = nextLayer;
        }
    }

    void teach(vector<double> input_data, vector<int> input_labels){
        vector<double> modelOutput = forwardPropagate(input_data);  
        //cout << "######################" << endl;
        cout << "loss "<<lossFunction (modelOutput, input_labels) << endl;
        vector<double> softmaxOutput = softmax(modelOutput);
        //cout << "######################" << endl;
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
        auto max_it = std::max_element(softmaxOutput.begin(), softmaxOutput.end()); //XXX max function can't handle equal inputs 
        int max_index = std::distance(softmaxOutput.begin(), max_it);
        if(input_labels[max_index] == 1) {
           correctPredictions += 1;  
           totalPredictions +=1;
        } else {
            totalPredictions +=1;
        }
        cout << "accuracy: " << float(correctPredictions)/float(totalPredictions) << endl;

        ofstream myfile;
          myfile.open ("accuracyLog.txt",ios::app);
          myfile << float(correctPredictions)/float(totalPredictions)  << "\n";
          myfile.close();

        cout << "######################" << endl;

        backPropagate(modelOutput, input_labels);
    }
    void infoLayers() {
        cout << "Layer (type)\t\tWeight Matrix Shape\t\tParam #" << endl;
        cout << "=======================================================================" << endl;

        Layer* current = topographyHead;
        int totalParams = 0;
        int numLayers = 0;

        while (current != NULL) {
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
    vector<double> forwardPropagate(vector<double> input_data){
     Layer* current = topographyHead;
    
     while (current != NULL) {
            input_data = current->forwardPropagate(input_data);
            current = current->next;
        }
    return input_data;
    }

void backPropagate(vector<double>& output, vector<int>& trueLabels) {
    vector<double> gradient = lossDerivative(output, trueLabels);

    std::vector<Layer*> layersToUpdate;
    Layer* current = topographyHead;
    while (current) {
        layersToUpdate.push_back(current);
        current = current->next;
    }
    std::reverse(layersToUpdate.begin(), layersToUpdate.end());

    for (Layer* layer : layersToUpdate) {
        layer->backwardPropagate(gradient, learningRate);
        gradient = layer->prevLayerGradient; 
    }
}

};




