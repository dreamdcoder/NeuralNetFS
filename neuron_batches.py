import numpy as np

def layer_output(inputs,weights,bias):
    x = np.array(inputs)
    w =np.array(weights)
    b = np.array(bias)
    print(f"inputs shape is {x.shape}")
    print(f"weights shape is {w.shape}")
    print(f"bias shape is {b.shape}")

    #x.wT+b
    #output is of form [input 1] -->  [neuron 1 ,neuron 2,neuorn3 ...]
    #                  [input 2] -->  [neuron 1 ,neuron 2,neuorn3 ...]
    layer_output= np.dot(x,w.T)+bias
    print(layer_output)

if __name__ == "__main__":
    inputs = [[1.0, 2.0, 3.0, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]

    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]

    biases = [2.0, 3.0, 0.5]
    layer_output(inputs=inputs,weights=weights,bias=biases)