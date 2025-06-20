import tensorflow as tf
import numpy as np

# Assumindo que 'x_train' e 'y_train' estão carregados (ex: via tf.keras.datasets.mnist.load_data())
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, axis=-1)

# Carregue seu modelo Keras salvo
model = tf.keras.models.load_model('mnist_cnn_small.tflite.h5')

# Crie um gerador para o dataset representativo
def representative_dataset_gen():
  for i in range(100): # Use ~100-500 amostras
    yield [x_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open('mnist_cnn_small_int8.tflite', 'wb') as f:
  f.write(tflite_quant_model)

# Gere o arquivo de cabeçalho C
# !xxd -i mnist_cnn_small_int8.tflite > mnist_model_data.h