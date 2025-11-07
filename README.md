# MLP_Backprop_Coursework


# MLP_Backprop_Coursework

This repository contains MATLAB implementations for a coursework on **incremental backpropagation for a 2-input, 3-hidden, 1-output Multilayer Perceptron (MLP)**. The coursework demonstrates both **manual step-by-step calculation** and a **generalized MATLAB prototype** for incremental backpropagation.

---

## **Repository Structure**



MLP_Backprop_Coursework/
│-- README.md
│-- Part1_MLP_ManualExample.m      # Hardcoded calculation example
│-- Part2_MLP_IncrementalGeneral.m # Generalized incremental backpropagation
│-- results/ (optional: logs, screenshots)



---

## **Part 1: Manual Example MLP (2x3x1)**

### **Purpose**
- To manually understand the **incremental backpropagation learning process**.  
- Tracks **forward pass outputs, hidden layer errors (betas), weight changes (ΔW), and updated weights**.  
- Demonstrates **two training samples**:
  1. `x1=0, x2=1, target=1`
  2. `x1=1, x2=0, target=1`  
- Uses **hard-coded weights (W1~W9)** for clarity and reproducibility.

### **Network Architecture**
- **Input layer:** 2 nodes (`x1`, `x2`)  
- **Hidden layer:** 3 sigmoid nodes (`h1`, `h2`, `h3`)  
- **Output layer:** 1 linear summation unit (no bias)  
- **Connections:**


W1*x1 → h1, W2*x2 → h1
W3*x1 → h2, W4*x2 → h2
W5*x1 → h3, W6*x2 → h3
W7*h1 → output, W8*h2 → output, W9*h3 → output



### **How It Works**
1. **Forward Pass:** Computes hidden layer outputs (sigmoid) and output layer value (linear).  
2. **Backward Pass:** Computes **beta values (errors)** for output and hidden nodes.  
3. **Weight Update:** Applies **incremental backpropagation**:


ΔW = learning_rate * beta * input

- Updates weights immediately after each sample.  
4. **Outputs Displayed:**  
- Hidden layer outputs  
- Output value  
- Beta values (errors)  
- Weight changes and updated weights  

### **Usage**
1. Open `Part1_MLP_ManualExample.m` in MATLAB.  
2. Run the script.  
3. Observe **step-by-step calculations** for each training sample.

---

## **Part 2: Generalized Incremental Backpropagation MLP**

### **Purpose**
- Provides a **general MATLAB prototype** for incremental backpropagation MLPs.  
- Supports **any number of input, hidden, and output nodes**.  
- Uses **random normal distribution initialization** (`randn`) for weights.  
- Automates forward pass, backward pass, and weight updates for multiple training samples.

### **Network Architecture**
- Configurable via parameters:
```matlab
N_input = 2;    % number of input nodes
N_hidden = 3;   % number of hidden nodes
N_output = 1;   % number of output nodes
````

* Hidden layer uses **sigmoid activation**; output layer is **linear**.
* **Incremental learning**: weights updated immediately after each sample.

### **How It Works**

1. **Initialize Weights:** Random normal distribution with mean 0 and standard deviation `sigma`.
2. **Forward Pass:** Compute hidden layer outputs (sigmoid) and output layer values (linear).
3. **Backward Pass:** Compute **error derivatives (beta)** for output and hidden layers.
4. **Weight Update:** Incremental updates:

   ```matlab
   Weights_hidden_output = Weights_hidden_output + eta * beta_out * h_out';
   Weights_input_hidden = Weights_input_hidden + eta * beta_hidden * x';
   ```
5. **Outputs Displayed:** For each sample:

   * Hidden layer outputs
   * Output value
   * Beta values (errors)
   * ΔW and updated weights

### **Usage**

1. Open `Part2_MLP_IncrementalGeneral.m` in MATLAB.
2. Adjust parameters (`N_input`, `N_hidden`, `N_output`, `eta`) and training data as needed.
3. Run the script.
4. Observe console output for all intermediate values and weight updates.
5. Compare with Part 1 if using the same 2x3x1 network and training samples.

---

## **Notes**

* Both scripts are self-contained and do not require external dependencies.
* Weight updates and beta values are displayed **up to 4 decimal places** to match marking criteria.
* Part 2 can handle **any MLP size or dataset**, while Part 1 demonstrates manual calculation for learning comprehension.
* Part 2 reproduces Part 1 results when using the same network structure and initial weights.

---

## **References**

* Dave Touretzky, CMU “Neural Network MATLAB Examples”
* Nikolay Nikolaev, modified MATLAB code for sunspot example
* Standard incremental backpropagation theory

---

```


