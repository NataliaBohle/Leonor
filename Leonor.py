import streamlit as st
import numpy as np
import whisper
import sounddevice as sd
import torch
from googletrans import Translator
import time

# Parámetros
FS = 16000
DURACION_FRAGMENTO = 3  # segundos
UMBRAL_SONIDO = 300

@st.cache_resource
def cargar_modelo(nombre_modelo="tiny"):
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(nombre_modelo, device=dispositivo)

def grabar_audio(duracion=3):
    st.info("🎙️ Grabando audio...")
    grabacion = sd.rec(int(duracion * FS), samplerate=FS, channels=1, dtype='int16')
    sd.wait()
    return grabacion.flatten()

def procesar_audio(modelo, audio, idioma_origen=None, idioma_destino=None):
    audio = (audio / np.max(np.abs(audio))).astype(np.float32)
    resultado = modelo.transcribe(audio, language=idioma_origen) if idioma_origen else modelo.transcribe(audio)
    
    texto = resultado["text"].strip()
    idioma_detectado = resultado.get("language", idioma_origen)

    traduccion = ""
    if texto and idioma_destino:
        traductor = Translator()
        try:
            traduccion = traductor.translate(texto, src=idioma_detectado, dest=idioma_destino).text
        except Exception as e:
            traduccion = f"⚠️ Error de traducción: {e}"
    
    return texto, idioma_detectado, traduccion

# === Interfaz Streamlit ===

st.title("🔁 Transcriptor y Traductor en Vivo")
st.markdown("Transcribe tu voz y muestra traducción en pantalla limpia.")

with st.sidebar:
    idioma_origen = st.text_input("Idioma de origen (ej: es)", value="es")
    idioma_destino = st.text_input("Idioma destino (ej: en)", value="en")
    modelo_nombre = st.selectbox("Modelo Whisper", options=["tiny", "base", "small", "medium", "large"], index=0)
    duracion = st.slider("Duración de grabación (segundos)", min_value=1, max_value=10, value=3)

modelo = cargar_modelo(modelo_nombre)

if st.button("🎙️ Comenzar grabación"):
    audio = grabar_audio(duracion)
    texto, idioma_detectado, traduccion = procesar_audio(modelo, audio, idioma_origen, idioma_destino)

    st.subheader("🗣️ Transcripción")
    st.write(texto if texto else "No se detectó texto.")

    if idioma_destino:
        st.subheader("🔁 Traducción")
        st.write(traduccion if traduccion else "No se realizó traducción.")
