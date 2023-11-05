#include "model.cpp"
#include "loadcsv.cpp"
#include <iomanip>

using namespace std;


#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>

using namespace std;

// Assuming the Model, CallBackFunctionType, load_mnist_csv and one_hot_encode 
// are defined somewhere in the program.

int main(int argc, char* argv[])
{
    cout << "#################################################################" << endl;
    cout << "Generating Model" << endl;
    cout << "#################################################################" << endl;

    // Default values
    double learningRate = 0.00015666;
    string optimizerType = "sgd"; // supported: sgd, sgd_momentum, adagrad, rmsprop, adam
    string activationFunction = "sigmoid";
    string runNote = "sigmoid_256_sgd";

    // Parse command line arguments
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "-lr") == 0 && i + 1 < argc) {
            learningRate = atof(argv[++i]);
        } else if(strcmp(argv[i], "-opt") == 0 && i + 1 < argc) {
            optimizerType = argv[++i];
        } else if(strcmp(argv[i], "-act") == 0 && i + 1 < argc) {
            activationFunction = argv[++i];
        } else if(strcmp(argv[i], "-note") == 0 && i + 1 < argc) {
            runNote = argv[++i];
        } else {
            cerr << "Usage: " << argv[0] << " [-lr learning_rate] [-opt optimizer] [-act activation_function] [-note run_note]" << endl;
            return 1;
        }
    }

    Model test("cross_entropy", optimizerType, learningRate, runNote);

    vector<CallBackFunctionType> callbacks;
    test.addLayer(activationFunction, make_tuple(256, 784), callbacks);
    test.addLayer(activationFunction, make_tuple(10, 256), callbacks);
    test.infoLayers();

    cout << "#################################################################" << endl;
    cout << "Begining Training" << endl;
    cout << "#################################################################" << endl;

    vector<int> labelNums;
    vector<vector<int>> images;
    if (load_mnist_csv("mnist_train.csv", labelNums, images))
    {
        cout << "Loaded " << images.size() << " training examples." << endl;
    }
    else
    {
        cerr << "Failed to load training data." << endl;
        return 1;
    }

    vector<vector<int>> label_vec = one_hot_encode(labelNums, 10);
    
    // subset data for faster testing:
    //label_vec.resize(5000);
    //images.resize(5000);
    
    test.teach(label_vec, images, 100);

    cout << "train accuracy " << test.getLastAccuracy() << endl;

    cout << "#################################################################" << endl;
    cout << "Finished training, begining testing..." << endl;
    cout << "#################################################################" << endl;

    vector<int> test_labelNums;
    vector<vector<int>> test_images;
    if (load_mnist_csv("mnist_test.csv", test_labelNums, test_images))
    {
        cout << "Loaded " << test_images.size() << " testing examples." << endl;
    }
    else
    {
        cerr << "Failed to load test data." << endl;
        return 1;
    }

    vector<vector<int>> test_label_vec = one_hot_encode(test_labelNums, 10);
    /*
    for(size_t i = 0; i < test_labelNums.size(); ++i) {
        vector<double> tmp = cast_vector_to_double(test_images[i]);
        test.teach(tmp, test_label_vec[i]);
    }
    */

    test.predict(test_label_vec, test_images);

    cout << "test accuracy " << test.getLastAccuracy() << endl;

    return 0;
}
