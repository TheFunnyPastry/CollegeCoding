/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Block.h"
#include <vector>

class NeuralNetwork {
private:
    std::vector<Block> blocks; //vector of blocks that make up the network

public:
    //constructor
    NeuralNetwork(int num_blocks, int in_size, int hidden_size, int out_size);

    //forward pass through all blocks
    std::vector<double> forward(const std::vector<double>& input) const;

    //set weights for a specific block's linear layer
    void setBlockWeights(int block_index, const std::vector<double>& flatWeights);

    //set activation weights for a specific block's ReLU layer
    void setBlockActivationWeights(int block_index, const std::vector<double>& actWeights);

    //print detailed information for each block
    void printModel() const;

    //print size information for each block
    void printBlockSizes() const;
};

#endif