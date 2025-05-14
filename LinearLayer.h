/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#define LINEARLAYER_H

#include "Module.h"
#include <vector>

class LinearLayer : public Module {
private:
    int input_size;  //number of inputs
    int output_size; //number of outputs
    std::vector<double> bias; //bias vector

public:
    //constructor
    LinearLayer(int in_size, int out_size);

    //forward pass: y = Wx + b
    std::vector<double> forward(const std::vector<double>& input) const override;

    //set flat weights
    void setFlatWeights(const std::vector<double>& newWeights);

    //display layer details
    void display() const;

    //print input and output sizes
    void size() const;
};