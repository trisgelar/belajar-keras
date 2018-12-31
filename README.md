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

