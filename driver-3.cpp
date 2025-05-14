#include "NeuralNetwork.h"
#include <iostream>
#include <vector>
using namespace std;
int main() {
// Create a neural network with 3 blocks.
// Architecture:
// First block: maps input of size 3 to hidden layer of size 4.
// Intermediate block(s): maps hidden layer (size 4) to hidden layer (size 4).
// Last block: maps hidden layer (size 4) to output of size 2.
NeuralNetwork nn(3, 3, 4, 2);
// Optionally, set custom flat weights for the first block's linear layer.
// Expecting 3 (in_size) * 4 (hidden_size) = 12 values.
vector<double> flatWeights = {
0.5, 0.6, 0.7, // Weights for first output neuron
0.8, 0.9, 1.0, // Weights for second output neuron
1.1, 1.2, 1.3, // Weights for third output neuron
1.4, 1.5, 1.6 // Weights for fourth output neuron
};
nn.setBlockWeights(0, flatWeights);
// Set custom activation weights (slope and negative slope) for the first block's ReLU layer.
vector<double> activationWeights = { 2.0, 0.1 };
nn.setBlockActivationWeights(0, activationWeights);
// Define an input vector of size 3.
vector<double> input = { 1.0, 0.5, -1.0 };
// Compute the network output with a forward pass.
vector<double> output = nn.forward(input);
// Display the network output.
cout << "Network output:" << endl;
for (double o : output) {
cout << o << " ";
}
cout << "\n\n";
// Display detailed model information.
nn.printModel();
nn.printBlockSizes();
return 0;
}
