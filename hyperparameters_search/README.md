# ML_hyperparameters_search
## Introduction 
In machine learning, a hyperparameter is a parameter whose value is set before the learning process begins. By contrast, the values of other parameters like the weights of a neural network are derived via training. Examples of hyperparameters for neural networks include the number of layers, the number of neurons in each layer and the learning rate.

Different training algorithms require different hyperparameters. Some simple algorithms (such as ordinary least squares regression) require none. The optimal values for the hyperparameters depend also on the dataset. In this assignment, you will implement a search algorithm to find good values for a small set of hyperparameters of a neural network.

Differential Evolution (DE) is an efficient and high performing optimizer for real-valued functions. DE is the search technique that you are required to use in this assignment to solve the optimization problems. As DE is based on evolutionary computing, it performs well on multi-modal, discontinuous optimization landscapes. Another appealing feature of DE is that it is still applicable even if the function we try to optimize is not differentiable. This is often the case when some simulation is involved in the computation of the real-valued function being optimized. In these scenarios, a gradient descent approach is not applicable. DE performs a parallel search over a population of fixed size, where each population member is a higher dimensional vector.

## Dataset

Two files in Dataset document:
* dataset_inputs.txt
* dataset_targets.txt

## Task
Adapting DE to perform a search for four hyperparameters of a neual network. 

1. Task 1
Fit a ploynomial to a noisy dataset and using DE.
2. Task 2
Perform a search for some hyperparameters of a neural network using DE
3. Task 3 
Run experiments to compare the following (population_size, max_iter) allocations in the list [(5,40), (10,20),(20,10),(40,5)]
