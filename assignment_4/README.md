# Assignment 4

Back-propagation neural network + perceptron for Assignment 4.

## Command Line Usage

- mode `enum` -> Mode of operation
  - manual
    - dataset `enum` -> Data set to train on
      - digits
      - faces
    - training_data_percent `f64` -> Percent of training data to use for training
      - **Ex.** 94.5 = 94.5%
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
  - sampler -> Automatically collects data using random samples of a given percent of training data
    - sample_count `usize` -> Number of samples to test for a given % of training data
    - dataset `enum` -> Data set to train on
      - digits
      - faces
    - training_data_percent `f64` -> Percent of training data to use for training
      - **Ex.** 94.5 = 94.5%
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
  - sampler_set -> Automatically collects data using random samples of [10%, 20%, 30%, ... 100%] of the training data
    - sample_count `usize` -> Number of samples to test for a given % of training data
    - dataset `enum` -> Data set to train on
      - digits
      - faces
    - training_data_percent `f64` -> Percent of training data to use for training
      - **Ex.** 94.5 = 94.5%
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
  - sampler_set -> Automatically collects data using random samples of training data for 10%, 20%, 30%, ... 100% of training data
    - sample_count `usize` -> Number of samples to test for a given % of training data
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
cargo run --release manual 100 digits neural 1234 "100 10" 20 0.01

📄 Neural network accuracy: 89.70%
```

**Faces (3 Layer)**
```bash
cargo run --release manual faces 100 neural 1234 "100 10" 20 0.01

📄 Neural network accuracy: 78.67%
```

**Digits (2 Layer)**
```bash
cargo run --release manual digits 100 neural 1234 "100" 20 0.01

📄 Neural network accuracy: 90.50%

cargo run --release sampler_set 30 digits neural 1234 "100" 20 0.01

📝 Summary
   AI: Neural network
   Data: 1️⃣  Digits
   Sample count: 30
🎯 Measurements:
   10.00% of data > Accuracy: (Mean: 79.21%, Stdev: 0.90%) Error: (Mean: 20.79%, Stdev: 0.90%) Time: (Mean: 8.6735s, Stdev: 0.8755s)
   20.00% of data > Accuracy: (Mean: 83.15%, Stdev: 0.81%) Error: (Mean: 16.85%, Stdev: 0.81%) Time: (Mean: 14.7276s, Stdev: 2.6994s)
   30.00% of data > Accuracy: (Mean: 85.00%, Stdev: 0.53%) Error: (Mean: 15.00%, Stdev: 0.53%) Time: (Mean: 19.7802s, Stdev: 4.3566s)
   40.00% of data > Accuracy: (Mean: 86.75%, Stdev: 0.45%) Error: (Mean: 13.25%, Stdev: 0.45%) Time: (Mean: 24.0922s, Stdev: 5.8281s)
   50.00% of data > Accuracy: (Mean: 87.64%, Stdev: 0.43%) Error: (Mean: 12.36%, Stdev: 0.43%) Time: (Mean: 28.1836s, Stdev: 7.8986s)
   60.00% of data > Accuracy: (Mean: 88.10%, Stdev: 0.50%) Error: (Mean: 11.90%, Stdev: 0.50%) Time: (Mean: 31.6786s, Stdev: 9.7489s)
   70.00% of data > Accuracy: (Mean: 88.53%, Stdev: 0.46%) Error: (Mean: 11.47%, Stdev: 0.46%) Time: (Mean: 34.8036s, Stdev: 11.7708s)
   80.00% of data > Accuracy: (Mean: 88.79%, Stdev: 0.53%) Error: (Mean: 11.21%, Stdev: 0.53%) Time: (Mean: 37.5439s, Stdev: 13.6510s)
   90.00% of data > Accuracy: (Mean: 89.24%, Stdev: 0.38%) Error: (Mean: 10.76%, Stdev: 0.38%) Time: (Mean: 40.1272s, Stdev: 16.5481s)
   100.00% of data > Accuracy: (Mean: 89.29%, Stdev: 0.44%) Error: (Mean: 10.71%, Stdev: 0.44%) Time: (Mean: 42.2255s, Stdev: 18.5556s)
```

**Faces (2 Layer)**
```bash
cargo run --release manual faces 100 neural 1234 "100" 8 0.01

📄 Neural network accuracy: 77.33%

cargo run --release sampler_set 30 faces neural 1234 100 20 0.01

📝 Summary
   AI: Neural network
   Data: 😀 Faces
   Sample count: 30
🎯 Measurements:
   10.00% of data > Accuracy: (Mean: 56.02%, Stdev: 3.48%) Error: (Mean: 43.98%, Stdev: 3.48%) Time: (Mean: 12.9332s, Stdev: 0.3914s)
   20.00% of data > Accuracy: (Mean: 61.62%, Stdev: 3.27%) Error: (Mean: 38.38%, Stdev: 3.27%) Time: (Mean: 24.0357s, Stdev: 1.9125s)
   30.00% of data > Accuracy: (Mean: 64.73%, Stdev: 2.79%) Error: (Mean: 35.27%, Stdev: 2.79%) Time: (Mean: 33.8300s, Stdev: 3.9136s)
   40.00% of data > Accuracy: (Mean: 67.78%, Stdev: 3.16%) Error: (Mean: 32.22%, Stdev: 3.16%) Time: (Mean: 42.4230s, Stdev: 6.4317s)
   50.00% of data > Accuracy: (Mean: 70.58%, Stdev: 3.41%) Error: (Mean: 29.42%, Stdev: 3.41%) Time: (Mean: 50.5396s, Stdev: 9.1878s)
   60.00% of data > Accuracy: (Mean: 72.44%, Stdev: 3.00%) Error: (Mean: 27.56%, Stdev: 3.00%) Time: (Mean: 57.0013s, Stdev: 12.7517s)
   70.00% of data > Accuracy: (Mean: 74.07%, Stdev: 2.84%) Error: (Mean: 25.93%, Stdev: 2.84%) Time: (Mean: 62.0311s, Stdev: 17.4784s)
   80.00% of data > Accuracy: (Mean: 75.56%, Stdev: 3.04%) Error: (Mean: 24.44%, Stdev: 3.04%) Time: (Mean: 66.1159s, Stdev: 22.7924s)
   90.00% of data > Accuracy: (Mean: 76.80%, Stdev: 2.81%) Error: (Mean: 23.20%, Stdev: 2.81%) Time: (Mean: 69.3053s, Stdev: 28.0580s)
   100.00% of data > Accuracy: (Mean: 77.49%, Stdev: 2.98%) Error: (Mean: 22.51%, Stdev: 2.98%) Time: (Mean: 72.2341s, Stdev: 32.5746s)

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