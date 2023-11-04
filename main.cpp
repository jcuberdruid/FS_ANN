#include "model.cpp"
#include "loadcsv.cpp"
#include <iomanip>
// #include"lossFuncs.cpp"
// #include"activationFuncs.cpp"
// #include"optimizer.cpp"
using namespace std;

int main()
{

    cout << "#################################################################" << endl;
    cout << "Generating Model" << endl;
    cout << "#################################################################" << endl;

    // Model
    Optimizer opt;
    double learningRate = 0.000478752177;
    Model test("cross_entropy", opt, learningRate);
    vector<CallBackFunctionType> callbacks;
    test.addLayer("tanh", make_tuple(1000, 784), callbacks);
    test.addLayer("tanh", make_tuple(1000, 1000), callbacks);
    test.addLayer("tanh", make_tuple(10, 1000), callbacks);
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
    /*
    for(size_t i = 0; i < labelNums.size(); ++i) {
        vector<double> tmp = cast_vector_to_double(images[i]);
        test.teach(tmp, label_vec[i]);
    }
    */

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
