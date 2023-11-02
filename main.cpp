#include"model.cpp"
#include"lossFuncs.cpp"
#include"activationFuncs.cpp"
//#include"optimizer.cpp"
using namespace std;

int main() {


//model test 
FunctionType lossFunction = atALossTest;
Optimizer opt;
double learningRate = 666.666;
Model test(lossFunction, opt, learningRate);

//add layer 
ActivFunctionType act = sigmoid_act;
vector<CallBackFunctionType> callbla;
test.addLayer(act, 10, make_tuple(5, 10), callbla);
test.addLayer(act, 10, make_tuple(5, 10), callbla);
test.addLayer(act, 10, make_tuple(5, 10), callbla);
test.addLayer(act, 10, make_tuple(5, 10), callbla);
test.addLayer(act, 10, make_tuple(5, 10), callbla);

test.infoLayers();

return 0;
}
