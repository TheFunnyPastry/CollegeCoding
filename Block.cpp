/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#include "Block.h"
#include <iostream>
#include <vector>

using namespace std;

//constructor
Block::Block(int in_size, int out_size)
    : linear(in_size, out_size), relu() {}

//forward pass: Linear transformation followed by ReLU activation
vector<double> Block::forward(const vector<double>& input) const {
    //pass input through the linear layer
    vector<double> linear_output = linear.forward(input);

    // Pass the result through the ReLU activation
    return relu.forward(linear_output);
}

//set weights for the linear layer
void Block::setWeights(const vector<double>& flatWeights) {
    linear.setFlatWeights(flatWeights);
}

//set weights for the ReLU activation
void Block::setActivationWeights(const vector<double>& actWeights) {
    relu.setWeights(actWeights);
}

//display block details
void Block::display() const {
    linear.display();
    relu.display();
}

//print size information
void Block::size() const {
    linear.size();
}