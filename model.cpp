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
#include"optimizer.cpp"
#include <iomanip>
using namespace std;
using FunctionType = double(*)(double); //for function pointers 
using CallBackFunctionType = void(*)(Layer*);

class Model {
public: 
    FunctionType lossFunction;        
    Optimizer optimizer;
    double learningRate;
    //int batchSize; <- mayber not 
    //optimizer <- interface 
    //data <- either vector or a class of amorphous shaping     
    vector<int> labels;
    Layer* topographyHead = NULL; 
    vector<CallBackFunctionType> callBacks; 
    
    Model(FunctionType lossFunction, Optimizer optimizer, double learningRate) : lossFunction(lossFunction),  optimizer(optimizer), learningRate(learningRate){}
    void addLayer(ActivFunctionType activationFunction, size_t numBias, tuple<int, int> shapeWeights, vector<CallBackFunctionType> callbacks) {
        Layer* newLayer = new Layer(activationFunction, numBias, shapeWeights, callbacks);
        
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

    void teach(){}
    void infoLayers() {
        // Header for the summary
        cout << "Layer (type)\t\tOutput Shape\t\tParam #" << endl;
        cout << "=================================================" << endl;

        Layer* current = topographyHead;
        int totalParams = 0;
        int numLayers = 0;

        while (current != NULL) {
            // Assuming weight shape is a tuple where the first int is the number of neurons (output shape)
            // and the second int is the input shape (not explicitly stored in the Layer class as per your code).
            auto shapeWeights = make_tuple(current->weights.size(), current->weights.empty() ? 0 : current->weights[0].size());
            int layerParams = get<0>(shapeWeights) * get<1>(shapeWeights) + get<0>(shapeWeights); // Weights + Bias

            // Print the layer details
            cout << setw(10) << numLayers << " (Layer)\t\t" 
                      << "[" << get<0>(shapeWeights) << "," << get<1>(shapeWeights) << "]\t\t"
                      << layerParams << endl;

            totalParams += layerParams;
            numLayers++;
            current = current->next;
        }

        // Footer for the summary
        cout << "=================================================" << endl;
        cout << "Total params: " << totalParams << endl;
        cout << "Trainable params: " << totalParams << endl; // Assuming all params are trainable
        cout << "Non-trainable params: 0" << endl; // Adjust if you have non-trainable parameters
        cout << "Total layers: " << numLayers << endl;
        }
    
private:
    void forwardPropagate(){}  
    void backPropagate(){}
};




