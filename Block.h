/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#define BLOCK_H

#include "LinearLayer.h"
#include "ReLUActivation.h"
#include <vector>

class Block {
private:
    LinearLayer linear;       //linear layer component
    ReLUActivation relu;      //reLU activation component

public:
    //constructor
    Block(int in_size, int out_size);

    //forward pass: Linear transformation followed by ReLU activation
    std::vector<double> forward(const std::vector<double>& input) const;

    //set weights for the linear layer
    void setWeights(const std::vector<double>& flatWeights);

    //set weights for the ReLU activation
    void setActivationWeights(const std::vector<double>& actWeights);

    //display block details
    void display() const;

    //print size information
    void size() const;
};