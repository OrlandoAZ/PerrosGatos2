# ========================================
#  STREAMLIT APP - CLASIFICACIÓN (VERSIÓN ROBUSTA FINAL)
# ========================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image, ImageOps

# ========================================
# FUNCIÓN PARA CARGAR EL MODELO (CON CACHÉ Y ARQUITECTURA MANUAL)
# ========================================
@st.cache_resource
def load_my_model():
    # --- Recrear la Arquitectura del Modelo ---
    # Esto evita los errores de deserialización del archivo .h5 en Keras 3
    class_labels = ["gato", "perro"]
    input_shape = (224, 224, 3)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights=None, # No descargar pesos, los cargaremos desde el archivo local
        alpha=0.5
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(class_labels), activation='softmax')
    ])
    
    # --- Cargar únicamente los Pesos ---
    # Este es el paso clave. No cargamos el modelo, solo los pesos entrenados.
    model.load_weights('keras_modelset.h5', by_name=True)
    
    return model, class_labels

# Cargar el modelo y las etiquetas
try:
    model, class_labels = load_my_model()
except Exception as e:
    st.error(f"Error fatal al cargar el modelo: {e}")
    st.stop()

# ========================================
# FUNCIÓN DE CLASIFICACIÓN
# ========================================
def clasificar_imagen(image_object, model_to_predict):
    img_resized = ImageOps.fit(image_object, (224, 224), Image.Resampling.LANCZOS)
    img_array_resized = np.asarray(img_resized)
    normalized_image_array = (img_array_resized.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    pred = model_to_predict.predict(data)[0]
    return pred

# ========================================
# INTERFAZ DE STREAMLIT
# ========================================
st.title("🐶🐱 Clasificador de Perros y Gatos")
st.write("Sube una imagen y el modelo (entrenado con Teachable Machine) la clasificará.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen seleccionada", use_column_width=True)
    
    with st.spinner("Clasificando..."):
        pred = clasificar_imagen(image, model)
        predicted_class_index = np.argmax(pred)
        predicted_class_label = class_labels[predicted_class_index]
        predicted_probability = pred[predicted_class_index]
        
        color = "red" if predicted_class_label.lower() == "gato" else "green"
        
        message = f'<p style="color: {color}; font-size: 24px;">La imagen es un <b>{predicted_class_label}</b> con una probabilidad de {predicted_probability:.2%}</p>'
        st.markdown(message, unsafe_allow_html=True)




