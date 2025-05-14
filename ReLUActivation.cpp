/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#include "ReLUActivation.h"
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

//constructor
ReLUActivation::ReLUActivation(double slope, double negative_slope) {
    //initialize weights: first element is slope, second is negative slope
    weights = {slope, negative_slope};
}

//forward pass: Apply ReLU activation
vector<double> ReLUActivation::forward(const vector<double>& input) const {
    if (weights.size() < 2) {
        throw runtime_error("ReLUActivation requires two weights: slope and negative slope.");
    }

    double slope = weights[0];
    double negative_slope = weights[1];
    vector<double> output;

    for (double x : input) {
        if (x >= 0) {
            output.push_back(slope * x);
        } else {
            output.push_back(negative_slope * x);
        }
    }

    return output;
}

//set activation weights
void ReLUActivation::setWeights(const vector<double>& newWeights) {
    if (newWeights.size() != 2) {
        throw runtime_error("ReLUActivation requires exactly two weights: slope and negative slope.");
    }
    weights = newWeights;
}

//display activation parameters
void ReLUActivation::display() const {
    if (weights.size() < 2) {
        throw runtime_error("ReLUActivation weights are not properly initialized.");
    }

    cout << "ReLUActivation: slope = " << weights[0] << ", negative slope = " << weights[1] << endl;
}