import sys
import keras as K
import tensorflow as tf

py_ver = sys.version
k_ver = K.__version__
tf_ver = tf.__version__

print("Using Python Version {}".format(py_ver))
print("Using Keras Version {}".format(k_ver))
print("Using Tensorflow Version {}".format(tf_ver))