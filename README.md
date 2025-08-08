```
Harmful-Wasp-Species-Classifier-Illinois

Able to distinguish between harmful and harmless species of wasps found in Illinois
Takes .jpg .jpeg .png images and returns one of the species of wasps encompassed by the model
Trained on iNaturalist research-grade images
Input must be resized to 224 x 224 pixels

TensorFlow machine learning model capable of image recognition of:
European Hornet/Vespa Crabro
Bald-Faced Hornet/Dolichovespula Maculata
Asian Paper Wasp/Polistes Chinensis
Eastern Yellowjacket/Vespula Maculifrons
Eastern Cicada Killer/Sphecius Speciosus

Model Architecture: EfficientNetB0
Accuracy: 83.11% (test data)
Dataset: Research-grade wasp images (500 samples per class).
Split: 70% training, 15% validation, 15% testing
Optimizer: Adam
Batch Size: 8
Epochs: 30(early stopping)
Dropout: 0.4

Dependencies required for full model:
tensorflow>=2.10.0 
numpy>=1.21.0
Pillow>=9.0.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0

For prediction:

----------

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('best_wasp_model.h5')  # Replace with your model path

CLASS_NAMES = [
  "Asian Paper Wasp (Polistes chinensis)",
  "Bald-Faced Hornet (Dolichovespula maculata)",
  "Eastern Cicada Killer (Sphecius speciosus)",
  "Eastern Yellowjacket (Vespula maculifrons)",
  "European Hornet (Vespa crabro)"
]

def predict_wasp_species(img_path):
  """Predict wasp species from an image file."""
  img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
  img_array = image.img_to_array(img)
  img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Normalize
  img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
  
  predictions = model.predict(img_array)
  predicted_class = CLASS_NAMES[np.argmax(predictions)]
  confidence = np.max(predictions) * 100

  print(f"Predicted: {predicted_class} ({confidence:.1f}% confidence)")

predict_wasp_species("path_to_your_image.jpg")

----------

Dependencies required for prediction:
tensorflow>=2.10.0
numpy>=1.21.0
Pillow>=9.0.0
```
