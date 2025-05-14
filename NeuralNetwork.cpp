/*
Name: William Brandon
FSUID : WMB22E
Section: 15
*/
#include "NeuralNetwork.h"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

NeuralNetwork::NeuralNetwork(int num_blocks, int in_size, int hidden_size, int out_size) {
    if (num_blocks < 1) {
        throw runtime_error("NeuralNetwork must have at least one block.");
    }

    //create the first block (input size -> hidden size or output size if only one block)
    if (num_blocks == 1) {
        blocks.emplace_back(in_size, out_size);
    } else {
        blocks.emplace_back(in_size, hidden_size);

        //create intermediate blocks (hidden size -> hidden size)
        for (int i = 1; i < num_blocks - 1; ++i) {
            blocks.emplace_back(hidden_size, hidden_size);
        }

        //create the last block (hidden size -> output size)
        blocks.emplace_back(hidden_size, out_size);
    }

    //initialize weights for all blocks
    for (size_t i = 0; i < blocks.size(); ++i) {
        int block_input_size = (i == 0) ? in_size : hidden_size;
        int block_output_size = (i == blocks.size() - 1) ? out_size : hidden_size;

        //set default weights for the block
        std::vector<double> default_weights(block_input_size * block_output_size, 1.0);
        blocks[i].setWeights(default_weights);
    }
}

//forward pass through all blocks
vector<double> NeuralNetwork::forward(const vector<double>& input) const {
    vector<double> output = input;

    //pass the input through each block sequentially
    for (const auto& block : blocks) {
        output = block.forward(output);
    }

    return output;
}

//set weights for a specific block's linear layer
void NeuralNetwork::setBlockWeights(int block_index, const std::vector<double>& flatWeights) {
    if (block_index < 0 || block_index >= static_cast<int>(blocks.size())) {
        throw std::runtime_error("Invalid block index.");
    }
    blocks[block_index].setWeights(flatWeights);
}

//set activation weights for a specific block's ReLU layer
void NeuralNetwork::setBlockActivationWeights(int block_index, const std::vector<double>& actWeights) {
    if (block_index < 0 || block_index >= static_cast<int>(blocks.size())) {
        throw std::runtime_error("Invalid block index.");
    }
    blocks[block_index].setActivationWeights(actWeights);
}

//print detailed information for each block
void NeuralNetwork::printModel() const {
    for (size_t i = 0; i < blocks.size(); ++i) {
        cout << "Block " << i + 1 << ":" << endl;
        blocks[i].display();
        cout << "-----------------------------" << endl;
    }
}

//print size information for each block
void NeuralNetwork::printBlockSizes() const {
    for (size_t i = 0; i < blocks.size(); ++i) {
        cout << "Block " << i + 1 << " size: ";
        blocks[i].size();
        cout << endl;
    }
}