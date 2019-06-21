<<<<<<< HEAD
# Siamese_Network

# Introduction
Despite impressive results in object classification, verification and recognition, most deep neural network based recognition systems become brittle when the view point of the camera changes dramatically. Robustness to geometric transformations is highly desirable for applications like wild life monitoring where there is no control on the pose of the objects of interest. The images of different objects viewed from various observation points define equivalence classes where by definition two images are said to be equivalent if they are views from the same object.
These equivalence classes can be learned via embeddings that map the input images to high dimensional vectors. During training, equivalent images are mapped to vectors that get pulled closer together, whereas if the images are not equivalent their associated vectors get pulled apart.

The propse of this project is to implement a deep neural network classifer to predict whether two handwritten images belong to the same class.The handwritten images are from the Keras’s digital image dataset – MNIST. The network used in this project is Siamese network The Siamese network has two identical subnetworks that shared the same weights followed by a distance calculation layer. 


# Dataset

import dataset from keras
```
keras.datasets.mnist.load_data()

``` 

* The network classifer will only learn the classification from digtis in [2,3,4,5,6,7]
* The digits in [0,1,8,9] are only used for testing. None of these digits should be used during training.

# Evaluation

* testing it with pairs from the set of digits [2,3,4,5,6,7]
* testing it with pairs from the set of digits [2,3,4,5,6,7] union [0,1,8,9] 
* testing it with pairs from the set of digits [0,1,8,9]

# Result 

Figure below indicates the model accuracy on training data and validation data. As can be seen from the figure, the model accuracy on training set increases significantly in the first few epochs, after around 15 epochs, model accuracy increases slightly and reaching 99.993%. The model accuracy on the validation set increased significantly in the first few epochs, after that it starts floating and did not improve much. After 100 epochs, the accuracy on the validation set is 0.9816%. Therefor checkpoint function here will save the model with the best accuracy on the validation set and drop the later model trained with 100 epochs.

![Image of acc](pic/acc.png)

Figure below indicates model loss on training data and validation data. As can be seen from the figure, the model loss on training set decreases significantly in first 20 epochs, after that model loss decreases slightly and reach 0.0136. The model loss on validation set decreases significantly in the first 20 epochs. After that, it starts floating and ends with 0.0306 after 100 epochs

![Image of acc](pic/loss.png)

Model accuracy on training set and validation

| Group         | Accuracy          |
| ------------- |:-------------:| 
| Training set  | 99.96%        |
| Testing set [2,3,4,5,6,7]    | 98.15%     |
| Testing set [0,1,8,9] | 70.19%     |
| Testing set [0,1,8,9] union [2,3,4,5,6,7]  | 83.67%       |
=======
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
>>>>>>> ML_hyperparameters_search/master
