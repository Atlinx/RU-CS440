# Assignment 4

Back-propagation neural network + perceptron for Assignment 4.

## Command Line Usage

- mode `enum` -> Mode of operation
  - manual
    - dataset `enum` -> Data set to train on
      - digits
      - faces
    - type `enum` -> Type of AI to use
      - neural
        - seed `u64` -> Random seed
        - training_data_percent `f64` -> Percent of training data to use for training
          - **Ex.** 94.5 = 94.5%
        - intermediate layers `String` -> Layers in neural network
          - String of `u64` separate by spaces
            - Each `u64` represents number of nodes (neurons) in a given layer
          - **Ex.** "100 10"
            - Produces neural network with input layer -> 100 nodes -> 10 nodes -> output layer
        - epochs `usize` -> Number of iterations trained on the same training data set
        - learn_rate `f64` -> Learning rate of the neural network
      - perceptron 
        - seed `u64` -> Random seed
        - training_data_percent `f64` -> Percent of training data to use for training
          - **Ex.** 94.5 = 94.5%
        - epochs `usize` -> Number of iterations trained on the same training data set
        - learn_rate `f64` -> Learning rate of the neural network
  - auto -> Automatically collects data using random samples of 10%, 20%, 30%, etc. of training data
    - samples `usize` -> Number of samples to test for a given % of training data
    - dataset `enum` -> Data set to train on
      - digits
      - faces
    - type `enum` -> Type of AI to use
      - neural
        - seed `u64` -> Random seed
        - intermediate layers `String` -> Layers in neural network
          - String of `u64` separate by spaces
            - Each `u64` represents number of nodes (neurons) in a given layer
          - **Ex.** "100 10"
            - Produces neural network with input layer -> 100 nodes -> 10 nodes -> output layer
        - epochs `usize` -> Number of iterations trained on the same training data set
        - learn_rate `f64` -> Learning rate of the neural network
      - perceptron 
        - seed `u64` -> Random seed
        - epochs `usize` -> Number of iterations trained on the same training data set
        - learn_rate `f64` -> Learning rate of the neural network

## Best Results

### Neural Network

**Digits (3 Layer)**
```bash
cargo run --release manual digits neural 1234 100 "100 10" 20 0.01

📄 Neural network accuracy: 89.70%
```

**Faces (3 Layer)**
```bash
cargo run --release manual faces neural 1234 100 "100 10" 20 0.01

📄 Neural network accuracy: 78.67%
```

**Digits (2 Layer)**
```bash
cargo run --release manual digits neural 1234 100 "100" 20 0.01

📄 Neural network accuracy: 90.50%
```

**Faces (2 Layer)**
```bash
cargo run --release manual faces neural 1234 100 "100" 8 0.01

📄 Neural network accuracy: 77.33%
```

### Perceptron

**Digits**
```bash
cargo run --release manual digits perceptron
```

**Faces**
```bash
cargo run --release manual faces perceptron
```