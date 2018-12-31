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
	print("IMDB sentiment analysis with keras")
	np.random.seed(1986)
	tf.set_random_seed(2018)

	# 1. load data training dan testing
	max_review_len = 50
	print("Loading train dan test data, max len = %d words\n" % max_review_len)

	train_x = np.loadtxt(
		'data/imdb_train_50w.txt', 
		usecols=range(0,max_review_len),
		delimiter=' ',
		dtype=np.float32
	)

	train_y = np.loadtxt(
		'data/imdb_train_50w.txt',
		usecols=[max_review_len],
		delimiter=' ',
		dtype=np.float32
	)

	test_x = np.loadtxt(
		'data/imdb_test_50w.txt',
		usecols=range(0,max_review_len),
		delimiter=' ',
		dtype=np.float32
	)

	test_y = np.loadtxt(
		'data/imdb_test_50w.txt', 
		usecols=[max_review_len],
		delimiter=' ',
		dtype=np.float32
	)

	# 2. Mendefinisikan Model
	e_init = K.initializers.RandomUniform(-0.01, 0.01, seed=1)
	init = K.initializers.glorot_uniform(seed=1)
	simple_adam = K.optimizers.Adam()
	nw = 129892
	embed_vec_len = 32

	model = K.models.Sequential()
	model.add(
		K.layers.embeddings.Embedding(
			input_dim=nw, 
			output_dim=embed_vec_len, 
			embeddings_initializer=init, 
			mask_zero=True)
	)
	model.add(
		K.layers.LSTM(
			units=100,
			kernel_initializer=init, 
			dropout=0.2)	
	)
	model.add(
		K.layers.Dense(
			units=1,
			kernel_initializer=init, 
			activation='sigmoid')
	)
	model.compile(
		loss='binary_crossentropy',
		optimizer=simple_adam,
		metrics=['acc']
	)
	print(model.summary())

	# 3. Train model
	b_size = 10
	max_epochs = 5
	
	print("training dimulai")
	h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, verbose=1, shuffle=True)
	print("training selesai")

	# 4. Evaluasi Model
	loss_acc = model.evaluate(test_x, test_y, verbose=0)
	print("Test data: loss = %0.6f accuracy = %0.2f%% " % (loss_acc[0], loss_acc[1]*100))

	# 5. Model di Simpan
	print("Model di simpan ke dalam disk")
	mp = 'model/imdb_model.h5'
	model.save(mp)

	print("Load Model dari disk")
	mp = 'model/imdb_model.h5'
	model = K.models.load_model(mp)

	# 6. Gunakan Model
	np.set_printoptions(precision=4)
	print("Sentiment for \"the movie was a great waste of my time\"")
	rev = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 20, 16, 6, 86, 425, 7, 58, 64]], dtype=np.float32)
	prediction = model.predict(rev)
	print("Prediction (0 = negative, 1 = positive) = ", end="")
	print("%0.4f" % prediction[0][0])

if __name__ == '__main__':
	main()