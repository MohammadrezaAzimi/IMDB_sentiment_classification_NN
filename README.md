# IMDB_sentiment_classification_NN

# Project description: In this project, there are 25000 highly polar movie reviews to construct training and development (hold-out) sets. The first 10000 reviews are kept as the development set while the rest is used for training set. There is an additional test set of 25000 reviews.

# Network model: A three-layer neural network is used. The network is comprised of two fully-connected layers with 16 hidden units and rectified linear unit (ReLU) activation as well as the output layer with sigmoid activation that classifies the reviews as positive or negative.

# Details: Optimization of stochastic gradient descent algorithm is done using RMSProp. Accuracy is performance metric and binary cross_entropy is the loss function.

1. Only 10000 most frequently words in the training set are kept.

2. Using Python code imdb.get_word_index() words are mapped to integer indices. Then a dictionary is constructed using dict(). 

3. To prepare the data, integer indices are transformed to tensors using one-hot encoding.

4.  The batch size is equal to 512 and then fitted the training data to the model over 20 epochs. 

# Modification:
1. To combat the overfitting problem, L2 regularization is used with weight coefficient . The entire 25000 reviews are used for training and then the resulting trained model is evaluated on the test set of 25000 reviews. The accuracy for the test set is 0.92 after 4 epochs.

2. As another approach to overcome overfitting problem, dropout regularization is used. The training, development and test tests are defined in the same way as original problem. The probability that each unit is kept is equal to 0.5.

Interested reader is referred to the book Deep Learning with Python by Francois Chollet.




