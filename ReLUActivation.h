/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#define RELUACTIVATION_H

#include "Module.h"
#include <vector>

class ReLUActivation : public Module {
public:
    //constructor with default slope values
    ReLUActivation(double slope = 1.0, double negative_slope = 0.0);

    //forward pass: Apply ReLU activation
    std::vector<double> forward(const std::vector<double>& input) const override;

    //set activation weights (slope and negative slope)
    void setWeights(const std::vector<double>& newWeights);

    //display activation parameters
    void display() const;
};