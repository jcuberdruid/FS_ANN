# From Scratch Artificial Neural Network - Image Classification

This is my implementation of an [ANN](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) from scratch in C++. It was able to classify the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) with 93% accuracy. 


The user facing parts of the library center around the [Model class](https://github.com/jcuberdruid/FS_ANN/blob/main/model.cpp). During instantiation a model object can take a number of activation functions and optimizers, as well as the learning rate. 
After instantiation the networks topology is formed by adding successive layers to the model, the topology can be viewed with the model.infoLayers() method. Finally the model can be trained with model.teach() which takes a vector<int> for labels, and a vector< vector<int>> for the images. 
After training takes place the trained model can be be tested with the model.predict() method which takes the same parameters as the aforementioned model.teach() method. 

The available loss functions can be seen in the [LossFuncsFactory.cpp](https://github.com/jcuberdruid/FS_ANN/blob/main/lossFuncsFactory.cpp); at present the supported loss functions are cross_entropy and softmax.
The activation functions available can be found in [ActivationFuncsFactory.cpp](https://github.com/jcuberdruid/FS_ANN/blob/main/activationFuncsFactory.cpp), with sigmoid, RELU, and tanh supported. 
