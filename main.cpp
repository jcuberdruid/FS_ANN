#include"model.cpp"
//#include"lossFuncs.cpp"
//#include"activationFuncs.cpp"
//#include"optimizer.cpp"
using namespace std;

int main() {

//model test 
Optimizer opt;
double learningRate = 0.01;
    
Model test("cross_entropy", opt, learningRate);

vector<CallBackFunctionType> callbla;
//                              outputSize, inputSize
test.addLayer("tanh", make_tuple(4,4), callbla); 
test.addLayer("tanh", make_tuple(100,4), callbla); 
test.addLayer("tanh", make_tuple(4,100), callbla); 
test.addLayer("tanh", make_tuple(4,4), callbla); 

test.infoLayers();

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(-5, 10);

std::vector<double> test_vals = {dist(mt), dist(mt), dist(mt), dist(mt)};   
std::vector<int> labels(test_vals.size(), 0);

auto max_it = std::max_element(test_vals.begin(), test_vals.end());
int max_index = std::distance(test_vals.begin(), max_it);

labels[max_index] = 1;

for (int i = 0; i < 30000; i++) {
    test.teach(test_vals, labels);
}

return 0;
}
