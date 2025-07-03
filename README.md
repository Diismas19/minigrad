# Minigrad - A tool for automatic differentiation

MiniGrad is a small and lightweight automatic differentiation framework implemented in Python from scratch, inspired by libraries such as PyTorch and TensorFlow. The goal of this project is purely educational, aiming to demystify the workings of backpropagation and dynamic computation graphs that are the heart of modern neural networks.

I developed this code as a academic activity to complete the course of Aritifcial Neural Networks.

This is a simple tool, soon more usage and advanced examples and usage will be added.

# About the Project

This project implements a mechanism of automatic differetiation in reverse mode, or backpropagation. This allow to contruct complex mathematical expressions and evaluate the gradient of any node of the computation in relation to any node in a eficient way.

The mainly data structure is the `Tensor`, which entangle a `numpy` array, and keep track of the operation which created it, building up a graph of dynamical computation. When the `.backward()` method is called in the last node, the gradients are propagated through the hole graph, filling the `.grad` atribute from each tensor.

# How to use

You need `python 3.8+`. In your directory, clone the repository.
```
git clone https://github.com/Diismas19/minigrad
cd minigrad
```
I strongly suggest that you use a virtual enverioment.
```
python3 -m venv venv
source venv/bin/activate
```
Then install the requirements.
```
pip install -r requirements.txt
```
Then install the package.
```
pip install -e .
```
You can run the `examples.ipynb` and use the tool for yourself.

# Contact

Vitor Petri - vitorpetrisilva@gmail.com
Link: https://github.com/Diismas19/minigrad
