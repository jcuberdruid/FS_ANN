#include"model.cpp"
#include"loadcsv.cpp"
#include <iomanip> 
//#include"lossFuncs.cpp"
//#include"activationFuncs.cpp"
//#include"optimizer.cpp"
using namespace std;

int main() {



cout << "#################################################################" << endl;
cout << "Generating Model" << endl;
cout << "#################################################################" << endl;

// Model
Optimizer opt;
double learningRate = 0.01;
Model test("cross_entropy", opt, learningRate);
vector<CallBackFunctionType> callbacks;
test.addLayer("tanh", make_tuple(256, 784), callbacks); 
test.addLayer("tanh", make_tuple(10, 256), callbacks); 
test.infoLayers();


cout << "#################################################################" << endl;
cout << "Begining Training" << endl;
cout << "#################################################################" << endl;

    vector<int> labelNums;
    vector<vector<int>> images;
    if (load_mnist_csv("mnist_train.csv", labelNums, images)) {
        cout << "Loaded " << images.size() << " training examples." << endl;
    } else {
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

    test.teach(label_vec, images, 10);

cout << "train accuracy " << test.getLastAccuracy() << endl;

cout << "#################################################################" << endl;
cout << "Finished training, begining testing..." << endl;
cout << "#################################################################" << endl;

/*
for (int i = 0; i < 300; i++) {
    test.predict(test_vals, test_labels);
}
cout << "test accuracy" << test.getLastAccuracy() << endl;
*/
return 0;
}
