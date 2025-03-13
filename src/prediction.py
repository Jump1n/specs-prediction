import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def predict_image(img_path, model_path="spectacles_model.h5"):
    """Load model, preprocess image, and predict glasses on/off"""
    
    model = keras.models.load_model(model_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  

    prediction = model.predict(img_array)[0][0]
    print(f"{prediction:.2f}")
    
    if prediction < 0.5:
        result = "With Glasses"
    else:
        result = "Without Glasses"
    
    plt.imshow(img)
    plt.title(f"Prediction: {result}")
    plt.axis("off")
    plt.show()

    print(f"🔍 Model Prediction: {result}")

predict_image("data/sample_test/images.jpeg")