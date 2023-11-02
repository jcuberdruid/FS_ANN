#include"model.cpp"
#include"lossFuncs.cpp"
//#include"activationFuncs.cpp"
//#include"optimizer.cpp"
using namespace std;

int main() {

//model test 
FunctionType lossFunction = atALossTest;
Optimizer opt;
double learningRate = 0.001;
    
Model test("cross_entropy", opt, learningRate);

//add layer 
vector<CallBackFunctionType> callbla;
test.addLayer("sigmoid", make_tuple(2,2), callbla); 
test.addLayer("sigmoid", make_tuple(2,2), callbla); 

//test forward propagation
test.infoLayers();

vector<double> test_vals(2, 1.0);
std::vector<int> labels(2, 0); 
labels[0] = 1;

for(int i = 0; i < 100000; i++){
    test.teach(test_vals, labels);
}

return 0;
}
