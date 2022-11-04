import numpy as np


def neuron_output(inputs, weights, bias):
    layer_output = []
    for neuron_weight, neuron_bias in zip(weights, bias):
        neuron_output = 0
        for neuron_input, weight in zip(inputs, neuron_weight):
            # Wi*Xi
            neuron_output += weight*neuron_input
            # Wi*Xi+b
        neuron_output += neuron_bias
        layer_output.append(neuron_output)
    return layer_output


def neuron_output_numpy(inputs, weights, bias):
    layer_output = np.dot(weights,inputs) + bias
    print(layer_output.shape)
    return layer_output


if __name__ == "__main__":
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    bias = [2, 3, 0.5]
    print(np.array(weights).shape,np.array(inputs).shape)
    # We can tell we have
    # three neurons because there are 3 sets of weights and 3 biases
    output = neuron_output(inputs=inputs, weights=weights, bias=bias)
    print(f"3 neuron's layer output is {output}")
    output = neuron_output_numpy(inputs=inputs, weights=weights, bias=bias)
    print(f"3 neuron's layer output using dot product is {output}")
