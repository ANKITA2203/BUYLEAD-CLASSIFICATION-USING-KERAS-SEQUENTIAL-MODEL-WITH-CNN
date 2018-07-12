# Buylead-Classification-using-Keras-Sequential-Model-with-CNN

Introduction

BUYLEADS: BuyLead is the purchase requirement filled in buylead forms by prospective buyers for products and services, which is filtered by IndiaMART and shared with the suppliers dealing in those products/services.By consuming a BuyLead, supplier gets access to buyer’s contact details.
This project of classifying buyleads comes under the domain of “Text Classification”,”Pattern analysis” and “Data Mining”. All these terms are very closely related and intertwined, and they can be formally defined as the process of discovering “useful” patterns in large set of data, either automatically (unsupervised) or semi automatically (supervised). The project would heavily rely on techniques of “Deep learning with Natural Language Processing” in extracting significant patterns and features from the large data set of  buyleads  and on “Machine Learning” techniques for accurately classifying individual labelled data samples (sold/unsold) according to whichever pattern model best describes them.

Classification can be broadly classified into three categories. These are Supervised classification, Unsupervised classification, and Semi-supervised document classification. In Supervised document classification, some mechanism external to the classification model (generally human) provides information related to the correct document classification. Thus, in case of Supervised document classification, it becomes easy to test the accuracy of classification model. In Unsupervised document classification, no information is provided by any external mechanism & In case of Semi-supervised document classification parts of the documents are labeled by an external mechanism.
 
This approach basically dividing buyleads into two different categories.It consists of two phases:
●	Training phase
●	Testing phase or Predicting phase


 The types of classification algorithms in Machine Learning:

1.	Linear Classifiers: Logistic Regression, Naive Bayes Classifier
2.	Support Vector Machines
3.	Decision Trees
4.	Boosted Trees
5.	Random Forest
6.	Neural Networks
7.	Nearest Neighbor

Classification algorithm used:
Neural Networks

Neural networks are well-suited to identifying non-linear patterns, as in patterns where there isn’t a direct, one-to-one relationship between the input and the output. Instead, the networks identify patterns between combinations of inputs and a given output.Neural networks are characterized by containing adaptive weights along paths between neurons that can be tuned by a learning algorithm that learns from observed data in order to improve the model.  In addition to the learning algorithm itself, one must choose an appropriate cost function. The cost function is what’s used to learn the optimal solution to the problem being solved. This involves determining the best values for all of the tunable model parameters, with neuron path adaptive weights being the primary target, along with algorithm tuning parameters such as the learning rate. It’s usually done through optimization techniques such as gradient descent or stochastic gradient descent.
These optimization techniques basically try to make the ANN solution be as close as possible to the optimal solution, which when successful means that the ANN is able to solve the intended problem with high performance.
Architecturally, an artificial neural network is modeled using layers of artificial neurons, or computational units able to receive input and apply an activation function along with a threshold to determine if messages are passed along.In a simple model, the first layer is the input layer, followed by one hidden layer, and lastly by an output layer. Each layer can contain one or more neurons.Models can become increasingly complex, and with increased abstraction and problem solving capabilities by increasing the number of hidden layers, the number of neurons in any given layer, and/or the number of paths between neurons. 

Model architecture and tuning are therefore major components of ANN techniques, in addition to the actual learning algorithms themselves. All of these characteristics of an ANN can have significant impact on the performance of the model.

There are several metrics proposed for computing and comparing the results of our experiments. Some of the most popular metrics include:Confusion Matrix.



Definition of the Terms:
• Positive (P): Observation is positive (for example: is an apple).
• Negative (N): Observation is not positive (for example: is not an apple).
• True Positive (TP): Observation is positive, and is predicted to be positive.
• False Negative (FN): Observation is positive, but is predicted negative.
• True Negative (TN): Observation is negative, and is predicted to be negative.
• False Positive (FP): Observation is negative, but is predicted positive.

Here,
• Class 1: Positive
• Class 2: Negative

 Precision, Recall, Accuracy, F1-measure, True rate and False alarm rate (each of these metrics  
 is calculated individually for each class and then averaged for the overall classifier.
 
2. DEFINITIONS

2.1 keras :
Keras is a minimalist Python library for deep learning that can run on top of Theano or TensorFlow.
It was developed to make implementing deep learning models as fast and easy as possible for research and development.
It runs on Python 2.7 or 3.5 and can seamlessly execute on GPUs and CPUs given the underlying frameworks. It is released under the permissive MIT license.
Keras was developed and maintained by François Chollet, a Google engineer using four guiding principles:

●	Modularity: A model can be understood as a sequence or a graph alone. All the concerns of a deep learning model are discrete components that can be combined in arbitrary ways.

●	Minimalism: The library provides just enough to achieve an outcome, no frills and maximizing readability.

●	Extensibility: New components are intentionally easy to add and use within the framework, intended for researchers to trial and explore new ideas.

●	Python: No separate model files with custom file formats. Everything is native Python.

 Keras can be installed easily using PyPI, as follows:
●	      sudo pip install keras

2.2 Sequential model:
The sequential API allows you to create models layer-by-layer for most problems.

2.3 Tokenizer class:
Keras provides the Tokenizer class for preparing text documents for deep learning. The
Tokenizer must be constructed and then fit on either raw text documents or integer encoded
text documents.

2.4 Baseline model:

A baseline is a method that uses heuristics, simple summary statistics, randomness, or
machine learning to create predictions for a dataset. You can use these predictions to
measure the baselines performance (e.g., accuracy)-- this metric will then become what
we compare any other machine learning algorithm against.

2.5 Training dataset:

The data we use is usually split into training data and test data. The training set contains a
 known output and the model learns on this data in order to be generalized to other data later on.
We have the test dataset (or subset) in order to test our model’s prediction on this subsets.
 
2.6 Test dataset:
A test dataset is a dataset that is independent of the training dataset, but that follows the same probability distribution as the training dataset.

2.7 Overfitting & Underfitting in model:

Overfitting occurs when a statistical model or machine learning algorithm captures the noise of the data.  Intuitively, overfitting occurs when the model or the algorithm fits the data too well.  Specifically, overfitting occurs if the model or algorithm shows low bias but high variance.  Overfitting is often a result of an excessively complicated model, and it can be prevented by fitting multiple models and using validation or cross-validation to compare their predictive accuracies on test data.

Underfitting occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data.  Intuitively, underfitting occurs when the model or the algorithm does not fit the data well enough.  Specifically, underfitting occurs if the model or algorithm shows low variance but high bias.  Underfitting is often a result of an excessively simple model.


2.8 BIAS & VARIANCE:

Bias refers to the error due to overly-simplistic assumptions or faulty assumptions in the learning algorithm. Bias results in under-fitting the data. A high bias means our learning algorithm is missing important trends amongst the features.


Variance refers to the error due to an overly-complex that tries to fit the training data as 
 closely as possible. In high variance cases the model’s predicted values are extremely close to the actual values from the training set. The learning algorithm copies the training data’s trends and this results in loss of generalization capabilities. High Variance gives rise to over-fitting.

2.9 FEATURE EXTRACTION:

Feature Selection in text classification refers to selecting a subset of the collection terms and utilize them in the process of text classification.
Good features are better indicators of a class label
Feature reduction tends to: – Reduce overfitting -- as it makes it less specific – Improve performance due to reducing dimensionality.
Feature Extraction provides more detailed features and feature relationships. 
