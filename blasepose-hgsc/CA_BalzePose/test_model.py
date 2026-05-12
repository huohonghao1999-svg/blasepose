import tensorflow as tf
from model import BlazePose

print("TensorFlow:", tf.__version__)

model = BlazePose()

x = tf.random.normal([1, 256, 256, 3])
y = model(x)

print("模型输出 shape:", y.shape)
print("模型参数量:", model.count_params())