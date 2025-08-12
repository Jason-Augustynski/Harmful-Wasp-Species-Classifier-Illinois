```
Harmful-Wasp-Species-Classifier-Illinois

Able to distinguish between harmful and harmless species of wasps found in Illinois
Takes .jpg .jpeg .png images and returns one of the species of wasps encompassed by the model
Trained on iNaturalist research-grade images
Input must be resized to 300 x 300 pixels

TensorFlow machine learning model capable of image recognition of:
- Cimbex americana (Elm Sawfly)  
- Dolichovespula arenaria (Aerial Yellowjacket)  
- Dolichovespula maculata (Bald-Faced Hornet)  
- Megacyllene robiniae (Locust Borer)  
- Milesia virginiensis (Yellowjacket Hover Fly)  
- Sphecius speciosus (Eastern Cicada Killer)  
- Sphex ichneumoneus (Great Golden Digger Wasp)  
- Sphex pensylvanicus (Great Black Wasp)  
- Tremex columba (Pigeon Tremex)  
- Vespa crabro (European Hornet)  
- Vespula germanica (German Yellowjacket)  
- Vespula maculifrons (Eastern Yellowjacket)  

Model Architecture: EfficientNetB0
  -512-unit Swish dense layer (L2 regularized)
  -Batch normalization
  -256-unit Swish dense layer
  -0.3 dropout rate
Training Procedure
  -Frozen Base: 15 epochs (learning rate=0.0001)
  -Fine Tuning: 25 epochs (laerning rate=1e-5)
Augmentations
  -Rotation (+=25/360)
  -Shear (+=10/360)
  -Zoom (+=20%)
  -Brightness adjustment (+=20%)
  -Channel shifts (+=50)
Accuracy: 87.44% (test data)
Dataset: Research-grade wasp images (1500 samples per class).
Split: 70% training, 15% validation, 15% testing
Optimizer: Adam
Batch Size: 8
Epochs: 30 (early stopping)

Dependencies required for full model:
tensorflow>=2.8.0 
numpy>=1.21.5
Pillow>=9.0.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
keras-preprocessing>=1.1.2

For prediction:

==========


#Python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

#Load model
model = tf.keras.models.load_model('best_wasp_model.h5')

CLASS_NAMES = [
    "Cimbex americana (Elm Sawfly)",
    "Dolichovespula arenaria (Aerial Yellowjacket)",
    "Dolichovespula maculata (Bald-Faced Hornet)",
    "Megacyllene robiniae (Locust Borer)",
    "Milesia virginiensis (Yellowjacket Hover Fly)",
    "Sphecius speciosus (Eastern Cicada Killer)",
    "Sphex ichneumoneus (Great Golden Digger Wasp)",
    "Sphex pensylvanicus (Great Black Wasp)",
    "Tremex columba (Pigeon Tremex)",
    "Vespa crabro (European Hornet)",
    "Vespula germanica (German Yellowjacket)",
    "Vespula maculifrons (Eastern Yellowjacket)"
]

def predict_wasp_species(img_path):
    """Predict wasp species from an image file."""
    img = image.load_img(img_path, target_size=(300, 300))  # Note: 300×300 resolution
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"Predicted: {predicted_class} ({confidence:.1f}% confidence)")

#Put file name here
predict_wasp_species("test_wasp.jpg")

==========

Dependencies required for prediction:
tensorflow>=2.8.0
numpy>=1.21.5
Pillow>=9.0.0
```
