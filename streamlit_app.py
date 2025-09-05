# ========================================
#  STREAMLIT APP - CLASIFICACIÓN (VERSIÓN FINAL CON PARCHE)
# ========================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.layers import DepthwiseConv2D # Importación necesaria para el parche

# ========================================
# PARCHE DE COMPATIBILIDAD para DepthwiseConv2D
# ========================================
# Este código intercepta la carga de la capa DepthwiseConv2D
# y elimina el argumento 'groups' que causa el error en algunas versiones de Keras.
try:
    original_from_config = DepthwiseConv2D.from_config
    @classmethod
    def patched_from_config(cls, config):
        config.pop('groups', None)
        return original_from_config(config)
    DepthwiseConv2D.from_config = patched_from_config
except AttributeError:
    # Si la versión de Keras es tan nueva que ya no tiene .from_config,
    # es posible que el parche no sea necesario. Lo ignoramos.
    pass

# ========================================
# FUNCIÓN PARA CARGAR EL MODELO (CON CACHÉ)
# ========================================
@st.cache_resource
def load_the_model():
    # Cargar el modelo. El parche anterior se encargará del error.
    model = tf.keras.models.load_model('keras_modelset.h5', compile=False)
    
    # Leer las etiquetas desde el archivo
    with open("labels.txt", "r") as f:
        class_labels = [line.strip().split(' ', 1)[1] for line in f]
        
    return model, class_labels

# Cargar el modelo y las etiquetas
try:
    model, class_labels = load_the_model()
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



