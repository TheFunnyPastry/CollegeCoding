/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#include "Module.h"
#include <iostream>
#include <vector>

using namespace std;

//default constructor
Module::Module() {
    weights = {1.0}; //initialize weights to default value
}

//forward pass: element-wise multiplication of input and weights
vector<double> Module::forward(const vector<double>& input) const {
    vector<double> output;
    for (size_t i = 0; i < input.size(); ++i) {
        output.push_back(input[i] * weights[i % weights.size()]);
    }
    return output;
}

//set weights
void Module::setWeights(const vector<double>& newWeights) {
    weights = newWeights;
}

//display weights
void Module::display() const {
    cout << "Weights: ";
    for (double w : weights) {
        cout << w << " ";
    }
    cout << endl;
}