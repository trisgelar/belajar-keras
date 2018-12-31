# ReadMe

## Belajar Keras dari Berbagai Sumber

1. Keras Succinctly [link](https://www.syncfusion.com/ebooks/keras-succinctly)


## Snippets

1. Import Keras
```
	from keras.models import Sequential
	from keras.layers import Dense, Activation
```

## Catatan
1. Common keras.initializers Functions

| Function 	| Description 	|
|-----------------------------------------------------	|-------------------------------------------------------------	|
| Zeros() 	| All np.float32 0.0 values 	|
| Constant(value=0)  	| All a single specified np.float32 value 	|
| RandomUniform(minval=-0.05, maxval=0.05, seed=None) 	| Random, evenly distributed between minval and maxval 	|
| glorot_normal(seed=None) 	| Truncated Normal with stddev = sqrt(2 / (fan_in + fan_out)) 	|
| glorot_uniform(seed=None) 	| Uniform random with limits sqrt(6 / (fan_in + fan_out)) 	|

2. Common Dense Layer Activation Functions

| Function 	| Description 	|
|------------------------------------	|-------------------------------------------------------------------------------------------	|
| relu(x, alpha=0.0, max_value=None) 	| if x < 0 , f(x) = 0, else f(x) = x 	|
| tanh(x) 	| hyperbolic tangent 	|
| sigmoid(x) 	| f(x) = 1.0 / (1.0 + exp(-x)) 	|
| linear(x) 	| f(x) = x 	|
| softmax(x, axis=-1) 	| coerces vector x values to sum to 1.0 so they can be loosely interpreted as probabilities 	|

3. Accuracy Metrics Functions

| Function 	| Description 	|
|--------------------------------------------------------	|----------------------------------	|
| binary_accuracy(y_true, y_pred) 	| For binary classification 	|
| categorical_accuracy(y_true, y_pred) 	| For multiclass classification 	|
| sparse_categorical_accuracy(y_true, y_pred) 	| Rarely used (see documentation)) 	|
| top_k_categorical_accuracy(y_true, y_pred, k=5) 	| Rarely used (see documentation)) 	|
| sparse_top_k_categorical_accuracy(y_true, y_pred, k=5) 	| Rarely used (see documentation)) 	|

4. Five Common Keras Optimizers

| Optimizer 	| Description 	|
|----------------------------------------------------------------------------------	|---------------------------------------------------------------------	|
| SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False) 	| Basic optimizer for simple neural networks 	|
| RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0) 	| Often used with recurrent neural networks, very similar to Adadelta 	|
| Adagrad(lr=0.01, epsilon=None, decay=0.0) 	| General purpose adaptive algorithm 	|
| Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0) 	| Advanced version of Adagrad, similar to RMSprop 	|
| Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) 	| Excellent general-purpose, adaptive algorithm 	|

5. Embedding Layer Parameters

| Name 	| Description 	|
|------------------------	|--------------------------------------------------------	|
| input_dim 	| Size of the vocabulary, i.e. maximum integer index + 1 	|
| output_dim 	| Dimension of the dense embedding 	|
| embeddings_initializer 	| Initializer for the embeddings matrix 	|
| embeddings_regularizer 	| Regularizer function applied to the embeddings matrix 	|
| embeddings_constraint 	| Constraint function applied to the embeddings matrix 	|
| mask_zero 	| Whether or not the input value 0 is a padding value 	|
| input_length 	| Length of input sequences, when it is constant 	|
