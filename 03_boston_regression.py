# Using Python Version 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)]
# Using Keras Version 2.2.4
# Using Tensorflow Version 1.9.0

import numpy as np
import keras as K
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class MyLogger(K.callbacks.Callback):
	def __init__(self, n, data_x, data_y, pct_close):
		self.n = n
		self.data_x = data_x
		self.data_y = data_y
		self.pct_close = pct_close

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.n == 0:
			curr_loss = logs.get('loss')
			total_acc = my_accuracy(self.model, self.data_x, self.data_y, self.pct_close)
			print("epoch = %4d curr batch loss (mse) = %0.6f accuracy = %0.2f%% \n" % (epoch, curr_loss, total_acc * 100))


def my_accuracy(model, data_x, data_y, pct_close):
	num_correct = 0; num_wrong = 0
	n = len(data_x)
	for i in range(n):
		predicted = model.predict(np.array([data_x[i]], dtype=np.float32))
		actual = data_y[i]
		if np.abs(predicted[0][0] - actual) < np.abs(pct_close * actual):
			num_correct += 1
		else:
			num_wrong += 1
	return (num_correct * 1.0) / (num_correct + num_wrong)


def main():
	# 0. mulai
	print("Boston houses dataset regression example")
	np.random.seed(1986)
	tf.set_random_seed(2018)

	# 1. load data training dan testing
	print("Loading Data boston kedalam memori")
	data_file = "data/boston_mm_tab.txt"
	all_data = np.loadtxt(
		data_file, 
		delimiter="\t", 
		skiprows=0,
		dtype=np.float32
	)

	N = len(all_data)
	indices = np.arange(N)
	np.random.shuffle(indices)
	n_train = int(0.80 * N)

	print("Spliting data kedalam training dan testing")
	data_x = all_data[indices,:-1]
	data_y = all_data[indices,-1]
	train_x = data_x[0:n_train,:]
	train_y = data_y[0:n_train]
	test_x = data_x[n_train:N,:]
	test_y = data_y[n_train:N]

	# 2. Mendefinisikan Model

	init = K.initializers.RandomUniform(seed=1)
	simple_sgd = K.optimizers.SGD(lr=0.010)
	model = K.models.Sequential()
	model.add(
		K.layers.Dense(
			units=10, 
			input_dim=13, 
			kernel_initializer=init, 
			activation='tanh')
	)
	model.add(
		K.layers.Dense(
			units=10,
			kernel_initializer=init, 
			activation='tanh')	
	)
	model.add(
		K.layers.Dense(
			units=1,
			kernel_initializer=init, 
			activation=None)
	)
	model.compile(
		loss='mean_squared_error',
		optimizer=simple_sgd,
		metrics=['mse']
	)

	# 3. Train model
	b_size = 8
	max_epochs = 500
	my_logger = MyLogger(int(max_epochs/5), train_x, train_y, 0.15)
	print("training dimulai")
	h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, verbose=0, callbacks=[my_logger])
	print("training selesai")

	# 4. Evaluasi Model
	acc = my_accuracy(model, train_x, train_y,  0.15)
	print("Overall accuracy (wthin 15%%) on training data = %0.4f" % acc)

	acc = my_accuracy(model, test_x, test_y,  0.15)
	print("Overall accuracy on test data = %0.4f \n" % acc)

	eval = model.evaluate(test_x, test_y, verbose=0)
	print("Overall loss (mse) on test data = %0.6f" % eval[0])

	# 5. Model di Simpan
	print("Model di simpan ke dalam disk")
	mp = 'model/boston_model.h5'
	model.save(mp)

	print("Load Model dari disk")
	mp = 'model/boston_model.h5'
	model = K.models.load_model(mp)

	# 6. Gunakan Model
	np.set_printoptions(precision=4)
	unknown = np.full(shape=(1,13), fill_value=0.6, dtype=np.float32)
	unknown[0][3] = -1.0 # binary feature
	predicted = model.predict(unknown)
	print("Using model to predict median house price for features: ")
	print(unknown)
	print("\nPredicted price is: ")
	print("$%0.2f" % (predicted * 10000))

if __name__ == '__main__':
	main()