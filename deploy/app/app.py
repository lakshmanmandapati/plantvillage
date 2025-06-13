import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Configuration ---
# Define model file paths and their corresponding class name file paths
# Ensure these match the actual names and locations in your 'models/' directory
MODEL_CONFIGS = [
    {"name": "Model 1 ", "model_path": "models/model_1.h5", "class_names_path": "models/class_names_1.txt"},
    {"name": "Model 2 ", "model_path": "models/model_2.h5", "class_names_path": "models/class_names_2.txt"},
    {"name": "Model 3 ", "model_path": "models/model_3.h5", "class_names_path": "models/class_names_3.txt"},
]

# --- Helper Functions ---

@st.cache_resource
def load_model_and_class_names(model_path, class_names_path):
    """
    Loads a Keras model and its associated class names from text file.
    Uses st.cache_resource to cache the model in memory, preventing
    reloading on every Streamlit rerun.
    """
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        
        # Load class names from the text file
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        st.success(f"Successfully loaded model from {model_path} and class names from {class_names_path}")
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model or class names from {model_path} / {class_names_path}: {e}")
        st.stop() # Stop the app if models can't be loaded

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses an image for model inference.
    Resizes the image and normalizes pixel values.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def make_prediction(model, image_array, class_names):
    """
    Performs inference using the loaded model and returns predictions.
    """
    predictions = model.predict(image_array)
    
    # Assuming classification models where output is probabilities for each class
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_probability = np.max(predictions, axis=1)[0]
    
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name, predicted_probability, predictions[0] # Return full probabilities for display

# --- Streamlit UI ---

st.set_page_config(
    page_title="Multi-Model Keras Deployment",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🤖 Multi-Model Keras Classifier")
st.markdown("Upload an image and let the models tell you what they see!")

# Load all models and class names at startup
models_and_classes = {}
for config in MODEL_CONFIGS:
    model, class_names = load_model_and_class_names(config["model_path"], config["class_names_path"])
    models_and_classes[config["name"]] = {"model": model, "class_names": class_names}

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.header("Predictions:")

    # Preprocess image once for all models
    # Assuming all models expect the same input size (e.g., 224x224)
    # Adjust target_size if your models require different input dimensions
    # For a robust solution, you might get model input shape dynamically:
    # input_shape = model.input_shape[1:3] # e.g., (224, 224)
    preprocessed_image = preprocess_image(image, target_size=(224, 224)) 

    # Iterate through each model and make predictions
    for model_name, data in models_and_classes.items():
        model = data["model"]
        class_names = data["class_names"]

        # Determine the target size for this specific model if they vary
        # For simplicity, we assume (224, 224) for all. If models have different
        # input shapes, you'd need to re-preprocess or adapt here.
        # Example: if model.input_shape is (None, H, W, C), target_size = (H, W)
        # current_model_target_size = model.input_shape[1:3]
        # preprocessed_image_for_current_model = preprocess_image(image, target_size=current_model_target_size)
        
        predicted_class_name, predicted_probability, all_probabilities = make_prediction(model, preprocessed_image, class_names)
        
        st.subheader(f"Results for {model_name}:")
        st.write(f"**Predicted Class:** `{predicted_class_name}`")
        st.write(f"**Confidence:** `{predicted_probability:.2f}`")

        # Optionally display top N predictions or all probabilities
        num_display_top = min(5, len(class_names)) # Display top 5 or fewer if less than 5 classes
        top_indices = np.argsort(all_probabilities)[::-1][:num_display_top]
        
        st.write("Top Predictions:")
        for i in top_indices:
            st.write(f"- {class_names[i]}: {all_probabilities[i]:.2f}")
        st.markdown("---")

else:
    st.info("Please upload an image to get predictions.")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit and TensorFlow by an ML Engineer.")

