# Using Python Version 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 23:09:28) [MSC v.1916 64 bit (AMD64)]
# Using Keras Version 2.2.4
# Using Tensorflow Version 1.9.0

import numpy as np
import keras as K
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
	# 0. mulai
	print("Iris dataset menggunakan keras dan tensorflow")
	np.random.seed(1986)
	tf.set_random_seed(2018)

	# 1. load data training dan testing
	print("Loading Data iris kedalam memori")
	train_file = "data/iris_train.txt"
	test_file = "data/iris_test.txt"

	train_x = np.loadtxt(
		train_file, 
		usecols=range(0,4),
		delimiter=",",
		skiprows=0,
		dtype=np.float32
	)
	train_y = np.loadtxt(
		train_file, 
		usecols=range(4,7),
		delimiter=",",
		skiprows=0,
		dtype=np.float32
	)

	test_x = np.loadtxt(
		test_file,
		usecols=range(0,4),
		delimiter=",",
		skiprows=0,
		dtype=np.float32
	)

	test_y = np.loadtxt(
		test_file,
		usecols=range(4,7),
		delimiter=",",
		skiprows=0,
		dtype=np.float32
	)

	# 2. Mendefinisikan Model

	init = K.initializers.glorot_uniform(seed=1)

	# Adam (adaptive moment estimation)
	simple_adam = K.optimizers.Adam()
	model = K.models.Sequential()
	model.add(
		K.layers.Dense(
			units=5, 
			input_dim=4, 
			kernel_initializer=init, 
			activation='relu')
	)
	model.add(
		K.layers.Dense(
			units=6,
			kernel_initializer=init, 
			activation='relu')	
	)
	model.add(
		K.layers.Dense(
			units=3,
			kernel_initializer=init, 
			activation='softmax')
	)
	model.compile(
		loss='categorical_crossentropy',
		optimizer=simple_adam,
		metrics=['accuracy']
	)

	# 3. Train model
	b_size = 1
	max_epochs = 10
	print("training dimulai")
	h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
	print("training selesai")

	print(h.history['loss'])

	# 4. Evaluasi Model
	eval = model.evaluate(test_x, test_y, verbose=0)
	print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1]*100))

	# 5. Model di Simpan
	print("Model di simpan ke dalam disk")
	mp = 'model/iris_model.h5'
	model.save(mp)

	print("Load Model dari disk")
	mp = 'model/iris_model.h5'
	model = K.models.load_model(mp)

	# 6. Gunakan Model
	np.set_printoptions(precision=4)
	unknown = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)
	predicted = model.predict(unknown)
	print("Using model to predict species for features: ")
	print(unknown)
	print("Predicted species is: ")
	print(predicted)

	labels = ["sentosa", "versicolor", "virginica"]
	idx = np.argmax(predicted)
	species = labels[idx]
	print(species)



if __name__ == '__main__':
	main()


