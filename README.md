# MiniGrad - An Automatic Differentiation Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg)

MiniGrad is a lightweight automatic differentiation engine implemented in Python from scratch, inspired by libraries such as PyTorch and TensorFlow. This project was developed as an academic activity for an Artificial Neural Networks course. Its goal is purely educational, aiming to demystify the workings of backpropagation and the dynamic computation graphs that are at the heart of modern neural networks.

This is a simple tool, and more advanced examples and use cases will be added soon.

---

## About the Project

This project implements the **reverse-mode automatic differentiation** mechanism, also known as backpropagation. It allows for the construction of complex mathematical expressions and the efficient calculation of the gradient of any node in the computation graph with respect to any other node.

The main data structure is the `Tensor`, which wraps a `numpy` array and keeps track of the operation that created it, dynamically building a computation graph. When the `.backward()` method is called on the final node (typically the loss function), the gradients are propagated throughout the entire graph, populating the `.grad` attribute of each tensor.

### Key Features

-   **Dynamic Computation Graph**: Operations are registered in real-time.
-   **Reverse-Mode Automatic Differentiation**: Efficiently calculates gradients via the `.backward()` method.
-   **Operator Overloading**: Use native Python operators (`+`, `*`, `@`, `**`) directly on `Tensor` objects.
-   **Library of Operations**: Includes common mathematical operations and activation functions (`ReLU`, `Sigmoid`, `Tanh`).

---

## Getting Started

To get the project up and running on your local machine, follow these steps.

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1.  **Clone the repository**
    ```sh
    git clone https://github.com/Diismas19/minigrad.git
    cd minigrad
    ```

2.  **Create and activate a virtual environment** (strongly recommended):
    ```sh
    # Create the environment
    python3 -m venv venv

    # Activate on Linux/macOS
    source venv/bin/activate

    # Activate on Windows
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Install the package in editable mode**:
    This makes the `minigrad` library importable in your scripts and notebooks.
    ```sh
    pip install -e .
    ```

You can now run the `examples.ipynb` notebook to see the tool in action. If you created a new environment, you may need to install `ipykernel` to make it available to Jupyter.

---

## Contact

Vitor Petri - vitorpetrisilva@gmail.com

Project Link: [https://github.com/Diismas19/minigrad](https://github.com/Diismas19/minigrad)