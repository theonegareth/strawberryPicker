import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('strawberry_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open('strawberry_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved as strawberry_model.tflite")