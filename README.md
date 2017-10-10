# Backpropagation using Numpy

Backpropagation, short for "backward propagation of errors", is an algorithm for supervised learning of artificial neural networks using gradient descent. Given an artificial neural network and an error function, the method calculates the gradient of the error function with respect to the neural network's weights. It is a generalization of the delta rule for perceptrons to multilayer feedforward neural networks.

Backpropagation's popularity has experienced a recent resurgence given the widespread adoption of deep neural networks for image recognition and speech recognition. It is considered an efficient algorithm, and modern implementations take advantage of specialized GPUs to further improve performance.

## backprop.py

Script creates two randomly initialized multilayer feedforward neural networks and iteratively updates weights of the first network via  backpropagation to match its output(s) with the second network.

### References

https://en.wikipedia.org/wiki/Backpropagation
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
http://neuralnetworksanddeeplearning.com/chap2.html
https://brilliant.org/wiki/backpropagation/
