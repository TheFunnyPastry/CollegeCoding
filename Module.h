/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#ifndef MODULE_H
#define MODULE_H

#include <vector>

class Module {
protected:
    std::vector<double> weights; //flat weight vector

public:
    //default constructor
    Module();

    //forward pass method
    virtual std::vector<double> forward(const std::vector<double>& input) const;

    //set weights
    void setWeights(const std::vector<double>& newWeights);

    //display weights
    void display() const;
};

#endif