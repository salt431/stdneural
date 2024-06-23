#include <iostream>
#include <vector>
#include <cmath>

// Neural network structure: 2 inputs, 4 hidden neurons, 3 outputs
const int INPUT_NEURONS = 2;
const int HIDDEN_NEURONS = 4;
const int OUTPUT_NEURONS = 3;

// Activation function: sigmoid
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Activation function derivative: for backpropagation
double sigmoidDerivative(double x) {
    return x * (1 - x);
}

class NeuralNetwork {
private:
    // Weight matrices
    std::vector<std::vector<double>> inputToHidden;
    std::vector<std::vector<double>> hiddenToOutput;

    // Biases
    std::vector<double> hiddenBias;
    std::vector<double> outputBias;

    // Weights for input-to-hidden connections (randomized for now)
    void initializeWeights() {
        for (int i = 0; i < INPUT_NEURONS; i++) {
            inputToHidden.emplace_back();
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                inputToHidden[i].push_back(rand() / (double)RAND_MAX - 0.5);
            }
        }
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            hiddenToOutput.emplace_back();
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                hiddenToOutput[i].push_back(rand() / (double)RAND_MAX - 0.5);
            }
        }
    }

    // Biases for hidden and output layers (randomized for now)
    void initializeBiases() {
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            hiddenBias.push_back(rand() / (double)RAND_MAX - 0.5);
        }
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            outputBias.push_back(rand() / (double)RAND_MAX - 0.5);
        }
    }

public:
    NeuralNetwork() {
        initializeWeights();
        initializeBiases();
    }

    // Train the network with a set of examples
    void train(std::vector<double> inputs, std::vector<double> expectedOutputs) {
        // Forward pass
        std::vector<double> hiddenOutputs(HIDDEN_NEURONS);
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            double sum = 0;
            for (int j = 0; j < INPUT_NEURONS; j++) {
                sum += inputs[j] * inputToHidden[j][i];
            }
            hiddenOutputs[i] = sigmoid(sum + hiddenBias[i]);
        }
        std::vector<double> outputOutputs(OUTPUT_NEURONS);
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            double sum = 0;
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                sum += hiddenOutputs[j] * hiddenToOutput[j][i];
            }
            outputOutputs[i] = sigmoid(sum + outputBias[i]);
        }

        // Backpropagation
        std::vector<double> hiddenErrors(HIDDEN_NEURONS);
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            double error = 0;
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                error += (expectedOutputs[j] - outputOutputs[j]) * hiddenToOutput[i][j] * sigmoidDerivative(hiddenOutputs[i]);
            }
            hiddenErrors[i] = error;
        }
        std::vector<double> outputErrors(OUTPUT_NEURONS);
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            double error = 0;
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                error += (expectedOutputs[j] - outputOutputs[j]) * sigmoidDerivative(outputOutputs[i]);
            }
            outputErrors[i] = error;
        }

        // Weight updates
        for (int i = 0; i < INPUT_NEURONS; i++) {
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                inputToHidden[i][j] += inputs[i] * hiddenErrors[j] * sigmoidDerivative(hiddenOutputs[j]);
            }
        }
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                hiddenToOutput[i][j] += hiddenOutputs[i] * outputErrors[j] * sigmoidDerivative(hiddenOutputs[i]);
            }
        }
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            hiddenBias[i] += hiddenErrors[i];
            outputBias[i] += outputErrors[i];
        }
    }

    // Make a prediction with the current network weights
    std::vector<double> predict(std::vector<double> inputs) {
        std::vector<double> hiddenOutputs(HIDDEN_NEURONS);
        for (int i = 0; i < HIDDEN_NEURONS; i++) {
            double sum = 0;
            for (int j = 0; j < INPUT_NEURONS; j++) {
                sum += inputs[j] * inputToHidden[j][i];
            }
            hiddenOutputs[i] = sigmoid(sum + hiddenBias[i]);
        }
        std::vector<double> outputOutputs(OUTPUT_NEURONS);
        for (int i = 0; i < OUTPUT_NEURONS; i++) {
            double sum = 0;
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                sum += hiddenOutputs[j] * hiddenToOutput[j][i];
            }
            outputOutputs[i] = sigmoid(sum + outputBias[i]);
        }
        return outputOutputs;
    }
};

int main() {
    NeuralNetwork network;

    // Train the network with some examples
    network.train({-1, -1}, {0, 0, 0});
    network.train({0, 0}, {1, 0, 0});
    network.train({1, 1}, {0, 1, 0});
    network.train({-1, 1}, {0, 0, 1});

    // Make a prediction
    std::vector<double> prediction = network.predict({0, 0});
    for (double output : prediction) {
        std::cout << output << " ";
    }
    std::cout << std::endl;

    return 0;
}
