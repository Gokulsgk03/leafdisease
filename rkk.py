import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model
@st.cache_resource
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Perform inference
def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    image = image.resize((224, 224))  # Resize to model's input size
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Load label file
@st.cache_data
def load_labels(label_path):
    try:
        with open(label_path, "r") as file:
            labels = [line.strip() for line in file.readlines()]
        return labels
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return []

# Main Streamlit app
def main():
    st.title("Leaf Disease Detection")
    st.write("Upload a leaf image, and the app will identify the disease present in the leaf.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    # Paths to model and labels
    model_path = r"C:\Users\gokul\Downloads\converted_tflite\model_unquant.tflite"
    label_path = r"C:\Users\gokul\Downloads\converted_tflite\labels.txt"

    # Load model and labels
    interpreter = load_tflite_model(model_path)
    labels = load_labels(label_path)

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

            if interpreter and labels:
                # Perform prediction
                output_data = predict(interpreter, image)
                predicted_index = np.argmax(output_data)
                predicted_label = labels[predicted_index]

                # Display the result
                st.success(f"The disease detected is: **{predicted_label}**")
            else:
                st.error("Model or labels not loaded properly.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
