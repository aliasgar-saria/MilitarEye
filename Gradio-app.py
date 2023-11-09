import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Load the trained model
model = load_model('model1.h5')  # Make sure 'model1.h5' is the correct path to your model

# Prediction function for the Gradio app
def predict_and_visualize(img):
    # Store the original image size
    original_size = img.size
    
    # Convert the input image to the target size expected by the model
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)
    
    # Assuming the model outputs a single-channel image, normalize to 0-255 range for display
    predicted_mask = (prediction[0, :, :, 0] * 255).astype(np.uint8)
    
    # Convert the prediction to a PIL image
    prediction_image = Image.fromarray(predicted_mask, mode='L')  # 'L' mode is for grayscale
    
    # Resize the predicted image back to the original image size
    prediction_image = prediction_image.resize(original_size, Image.NEAREST)

    return prediction_image

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil"),  # We expect a PIL Image
    outputs=gr.Image(type="pil"),  # We will return a PIL Image
    title="MilitarEye: Military Stealth Camouflage Detector",
    description="Please upload an image of a military personnel camouflaged in their surroundings. On the right, the model will attempt to predict the camouflage mask silhouette."
)

# Launch the Gradio app
iface.launch()
