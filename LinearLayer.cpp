/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#include "LinearLayer.h"
#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

//constructor
LinearLayer::LinearLayer(int in_size, int out_size)
    : input_size(in_size), output_size(out_size) {
    //initialize bias to 1.0 for each output
    bias = vector<double>(output_size, 1.0);
}

//forward pass: y = Wx + b
vector<double> LinearLayer::forward(const vector<double>& input) const {
    if (static_cast<int>(weights.size()) != input_size * output_size) {
        cerr << "Error in forward: Expected size = " << input_size * output_size
             << ", Actual size = " << weights.size() << endl;
        throw runtime_error("Weight vector size does not match input and output dimensions.");
    }

    vector<double> output(output_size, 0.0);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            output[i] += weights[i * input_size + j] * input[j];
        }
        output[i] += bias[i];
    }

    return output;
}

//set flat weights
void LinearLayer::setFlatWeights(const vector<double>& newWeights) {
    if (static_cast<int>(newWeights.size()) != input_size * output_size) {
        cerr << "Error in setFlatWeights: Expected size = " << input_size * output_size
             << ", Actual size = " << newWeights.size() << endl;
        throw runtime_error("New weight vector size does not match input and output dimensions.");
    }
    weights = newWeights;
}

//display layer details
void LinearLayer::display() const {
    cout << "LinearLayer expected weight dimensions: " << input_size << " (input) x " << output_size << " (output)" << endl;
    cout << "Bias: ";
    for (double b : bias) {
        cout << b << " ";
    }
    cout << endl;
    cout << "Flat weight vector: ";
    for (double w : weights) {
        cout << w << " ";
    }
    cout << endl;
}

//print input and output sizes
void LinearLayer::size() const {
    cout << "LinearLayer - Input size: " << input_size << ", Output size: " << output_size << endl;
}