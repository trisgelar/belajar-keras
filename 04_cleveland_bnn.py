# Using Python Version 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)]
# Using Keras Version 2.2.4
# Using Tensorflow Version 1.9.0

import numpy as np
import keras as K
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class MyLogger(K.callbacks.Callback):
	def __init__(self, n):
		self.n = n

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.n == 0:
			t_loss = logs.get('loss')
			t_accu = logs.get('acc')
			v_loss = logs.get('val_loss')
			v_accu = logs.get('val_acc')
			print("epoch = %4d t_loss = %0.4f t_acc = %0.2f%% v_loss = %0.4f v_acc = %0.2f%%" % (epoch, t_loss, t_accu*100, v_loss, v_accu*100))

def main():
	# 0. mulai
	print("Cleveland binary classifier example")
	np.random.seed(1986)
	tf.set_random_seed(2018)

	# 1. load data training dan testing
	print("Loading Cleveland kedalam memori")
	train_file = "data/cleveland_train.txt"
	valid_file = "data/cleveland_validate.txt"
	test_file = "data/cleveland_test.txt"

	train_x = np.loadtxt(
		train_file, 
		usecols=range(0,18),
		delimiter='\t',
		skiprows=0,
		dtype=np.float32
	)

	train_y = np.loadtxt(
		train_file, 
		usecols=[18],
		delimiter='\t',
		skiprows=0,
		dtype=np.float32
	)

	valid_x = np.loadtxt(
		valid_file, 
		usecols=range(0,18),
		delimiter='\t',
		skiprows=0,
		dtype=np.float32
	)

	valid_y = np.loadtxt(
		valid_file, 
		usecols=[18],
		delimiter='\t',
		skiprows=0,
		dtype=np.float32
	)

	test_x = np.loadtxt(
		test_file, 
		usecols=range(0,18),
		delimiter='\t',
		skiprows=0,
		dtype=np.float32
	)

	test_y = np.loadtxt(
		test_file, 
		usecols=[18],
		delimiter='\t',
		skiprows=0,
		dtype=np.float32
	)

	# 2. Mendefinisikan Model

	init = K.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=1)
	simple_adadelta = K.optimizers.Adadelta()
	X = K.layers.Input(shape=(18,))
	net = K.layers.Dense(
		units=10, 
		kernel_initializer=init,
		activation='relu'
	)(X)
	net = K.layers.Dropout(0.25)(net)
	net = K.layers.Dense(
		units=10,
		kernel_initializer=init,
		activation='relu'
	)(net)
	net = K.layers.Dropout(0.25)(net)
	net = K.layers.Dense(
		units=1,
		kernel_initializer=init,
		activation='sigmoid'
	)(net)
	model = K.models.Model(X, net)

	model.compile(
		loss='binary_crossentropy',
		optimizer=simple_adadelta,
		metrics=['acc']
	)

	# 3. Train model
	b_size = 8
	max_epochs = 2000
	my_logger = MyLogger(int(max_epochs/5))
	
	print("training dimulai")
	h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, verbose=0, validation_data=(valid_x, valid_y), callbacks=[my_logger])
	print("training selesai")

	# 4. Evaluasi Model
	eval = model.evaluate(test_x, test_y, verbose=0)
	print("Evaluation on test data: loss = %0.4f accuracy = %0.2f%% \n" % (eval[0], eval[1]*100))

	# 5. Model di Simpan
	print("Model di simpan ke dalam disk")
	mp = 'model/cleveland_model.h5'
	model.save(mp)

	print("Load Model dari disk")
	mp = 'model/cleveland_model.h5'
	model = K.models.load_model(mp)

	# 6. Gunakan Model
	np.set_printoptions(precision=4)
	unknown = np.array([[0.75, 1, 0, 1, 0, 0.49, 0.27, 1, -1, -1, 0.62, -1, 0.40, 0, 1, 0.23, 1, 0]], dtype=np.float32)
	unknown[0][3] = -1.0 # binary feature
	predicted = model.predict(unknown)
	print("Using model to predict heart disease for features: ")
	print(unknown)
	print("\nPredicted (0=no disease, 1=disease) is: ")
	print(predicted)



if __name__ == '__main__':
	main()