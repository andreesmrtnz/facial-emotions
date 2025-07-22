#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceMood AI - Descubre tu emoción, edad y género en tiempo real con inteligencia artificial
===============================================================================

Aplicación web interactiva que analiza emociones, edad y género usando DeepFace
y Streamlit con webcam en tiempo real.

Autor: FaceMood AI
Fecha: 2024
"""

import streamlit as st
import cv2
import numpy as np
import json
import time
from datetime import datetime
import os
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading
from deepface import DeepFace
import tempfile

# Configuración de la página
st.set_page_config(
    page_title="FaceMood AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración RTC para WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class FaceMoodTransformer(VideoTransformerBase):
    """Transformador de video para análisis en tiempo real."""
    
    def __init__(self):
        self.frame_count = 0
        self.analysis_interval = 30  # Aumentado para reducir lag
        self.current_results = {
            'emotion': 'neutral',
            'age': 25,
            'gender': 'unknown',
            'confidence': 0.0
        }
        self.emotion_history = []
        self.lock = threading.Lock()
        self.last_analysis_time = 0
        self.analysis_cooldown = 4.0  # 4 segundos entre análisis
        
        # Mapeo de emociones a emojis
        self.emotion_emojis = {
            'angry': '😠', 'disgust': '🤢', 'fear': '😨',
            'happy': '😀', 'sad': '😢', 'surprise': '😮', 'neutral': '😐'
        }
        
        # Colores para emociones (BGR)
        self.colors = {
            'angry': (0, 0, 255), 'disgust': (0, 255, 0), 'fear': (255, 0, 255),
            'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (255, 165, 0),
            'neutral': (128, 128, 128)
        }
    
    def recv(self, frame):
        """Procesa cada frame del video (método actualizado)."""
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        # Analizar cada cierto número de frames y con cooldown
        if (self.frame_count % self.analysis_interval == 0 and 
            current_time - self.last_analysis_time > self.analysis_cooldown):
            
            self.last_analysis_time = current_time
            
            try:
                # Crear archivo temporal con nombre único
                temp_filename = f"temp_frame_{int(current_time * 1000)}.jpg"
                temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
                
                # Guardar frame
                cv2.imwrite(temp_path, img)
                
                # Analizar con DeepFace
                result = DeepFace.analyze(
                    temp_path, 
                    actions=['emotion', 'age', 'gender'],
                    enforce_detection=False
                )
                
                if result and len(result) > 0:
                    analysis = result[0]
                    
                    # Extraer emociones
                    emotions = analysis.get('emotion', {})
                    dominant_emotion = max(emotions, key=emotions.get) if emotions else 'neutral'
                    emotion_confidence = emotions.get(dominant_emotion, 0) / 100.0
                    
                    # Extraer edad y género
                    age = analysis.get('age', 25)
                    gender = analysis.get('dominant_gender', 'unknown')
                    gender_confidence = analysis.get('gender', {}).get(gender, 0) / 100.0
                    
                    # Actualizar resultados
                    with self.lock:
                        self.current_results = {
                            'emotion': dominant_emotion,
                            'age': age,
                            'gender': gender,
                            'confidence': emotion_confidence,
                            'gender_confidence': gender_confidence,
                            'all_emotions': emotions
                        }
                        self.emotion_history.append(dominant_emotion)
                        if len(self.emotion_history) > 15:  # Reducido para mejor rendimiento
                            self.emotion_history.pop(0)
                    
                    # Actualizar session_state para la interfaz
                    st.session_state.current_emotion = dominant_emotion
                    st.session_state.current_confidence = emotion_confidence
                    st.session_state.current_age = age
                    st.session_state.current_gender = gender
                    st.session_state.current_gender_confidence = gender_confidence
                    st.session_state.emotion_history = self.emotion_history.copy()
                    st.session_state.last_update = current_time
                    st.session_state.analysis_count = st.session_state.get('analysis_count', 0) + 1
                    
                    print(f"🎭 Emoción detectada: {dominant_emotion} ({emotion_confidence:.2f}) - Edad: {age} - Género: {gender}")
                
                # Limpiar archivo temporal de forma segura
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    print(f"⚠️ No se pudo eliminar archivo temporal: {e}")
                    
            except Exception as e:
                print(f"Error en análisis: {e}")
        
        # Dibujar información en el frame
        img = self.draw_info_on_frame(img)
        
        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def draw_info_on_frame(self, img):
        """Dibuja información en el frame."""
        # Detectar rostros para dibujar rectángulos
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        with self.lock:
            current_results = self.current_results.copy()
        
        for (x, y, w, h) in faces:
            # Obtener color y emoji
            emotion = current_results['emotion']
            color = self.colors.get(emotion, (255, 255, 255))
            emoji = self.emotion_emojis.get(emotion, '❓')
            
            # Dibujar rectángulo
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            
            # Preparar texto
            emotion_text = f"{emoji} {emotion.upper()}"
            age_gender_text = f"Age: {current_results['age']} | {current_results['gender'].title()}"
            confidence_text = f"Conf: {current_results['confidence']:.2f}"
            
            # Configurar fuente
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Obtener tamaños de texto
            (emotion_width, emotion_height), _ = cv2.getTextSize(emotion_text, font, font_scale, thickness)
            (age_width, age_height), _ = cv2.getTextSize(age_gender_text, font, 0.5, 1)
            
            # Dibujar fondo
            cv2.rectangle(img, (x, y - emotion_height - 80), (x + max(emotion_width, age_width) + 15, y), color, -1)
            
            # Dibujar textos
            cv2.putText(img, emotion_text, (x + 5, y - 55), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(img, age_gender_text, (x + 5, y - 35), font, 0.5, (255, 255, 255), 1)
            cv2.putText(img, confidence_text, (x + 5, y - 15), font, 0.5, (255, 255, 255), 1)
        
        return img

def create_emotion_chart(emotion_history):
    """Crea gráfico de emociones con Plotly."""
    if not emotion_history:
        return None
    
    # Contar emociones
    emotion_counts = defaultdict(int)
    for emotion in emotion_history:
        emotion_counts[emotion] += 1
    
    # Crear gráfico de barras
    fig = px.bar(
        x=list(emotion_counts.keys()),
        y=list(emotion_counts.values()),
        title="Evolución de Emociones en Tiempo Real",
        labels={'x': 'Emociones', 'y': 'Frecuencia'},
        color=list(emotion_counts.keys()),
        color_discrete_map={
            'angry': '#FF0000', 'disgust': '#00FF00', 'fear': '#FF00FF',
            'happy': '#FFFF00', 'sad': '#0000FF', 'surprise': '#FFA500',
            'neutral': '#808080'
        }
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    """Función principal de la aplicación."""
    
    # Título y descripción
    st.title("🧠 FaceMood AI")
    st.markdown("### Descubre tu emoción, edad y género en tiempo real con inteligencia artificial")
    
    # Sidebar
    st.sidebar.title("🎛️ Controles")
    
    # Configuración
    analysis_interval = st.sidebar.slider(
        "Intervalo de análisis (frames)", 
        min_value=20, 
        max_value=60, 
        value=30,
        help="Cada cuántos frames se analiza la imagen (mayor = menos lag)"
    )
    
    # Estado de la aplicación
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = 'neutral'
    if 'current_confidence' not in st.session_state:
        st.session_state.current_confidence = 0.0
    if 'current_age' not in st.session_state:
        st.session_state.current_age = 25
    if 'current_gender' not in st.session_state:
        st.session_state.current_gender = 'unknown'
    if 'current_gender_confidence' not in st.session_state:
        st.session_state.current_gender_confidence = 0.0
    if 'last_update' not in st.session_state:
        st.session_state.last_update = 0
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'captures' not in st.session_state:
        st.session_state.captures = []
    
    # Crear transformador
    transformer = FaceMoodTransformer()
    transformer.analysis_interval = analysis_interval
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Webcam en Vivo")
        
        # WebRTC Streamer
        webrtc_ctx = webrtc_streamer(
            key="facemood",
            video_transformer_factory=lambda: transformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Mostrar resultados en tiempo real
        if webrtc_ctx.state.playing:
            st.subheader("🎭 Resultados en Tiempo Real")
            
            # Usar datos de session_state para mejor rendimiento
            current_emotion = st.session_state.get('current_emotion', 'neutral')
            current_confidence = st.session_state.get('current_confidence', 0.0)
            current_age = st.session_state.get('current_age', 25)
            current_gender = st.session_state.get('current_gender', 'unknown')
            current_gender_confidence = st.session_state.get('current_gender_confidence', 0.0)
            emotion_history = st.session_state.get('emotion_history', [])
            last_update = st.session_state.get('last_update', 0)
            analysis_count = st.session_state.get('analysis_count', 0)
            
            # Mapeo de emojis
            emotion_emojis = {
                'angry': '😠', 'disgust': '🤢', 'fear': '😨',
                'happy': '😀', 'sad': '😢', 'surprise': '😮', 'neutral': '😐'
            }
            
            # Mostrar métricas
            col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
            
            with col_metrics1:
                st.metric(
                    "Emoción", 
                    f"{emotion_emojis.get(current_emotion, '❓')} {current_emotion.title()}",
                    f"{current_confidence:.1%}"
                )
            
            with col_metrics2:
                st.metric(
                    "Edad", 
                    f"{current_age} años"
                )
            
            with col_metrics3:
                st.metric(
                    "Género", 
                    current_gender.title(),
                    f"{current_gender_confidence:.1%}"
                )
            
            with col_metrics4:
                st.metric(
                    "Análisis Realizados", 
                    analysis_count
                )
            
            # Mostrar emoción actual
            st.info(f"🎭 **Emoción detectada**: {emotion_emojis.get(current_emotion, '❓')} {current_emotion.title()} ({current_confidence:.1%})")
            
            # Mostrar última actualización
            if last_update > 0:
                time_since_update = time.time() - last_update
                if time_since_update < 5:
                    st.success(f"✅ Última actualización: hace {time_since_update:.1f} segundos")
                else:
                    st.warning(f"⚠️ Última actualización: hace {time_since_update:.1f} segundos")
            
            # Forzar actualización automática
            if st.button("🔄 Actualizar Datos", key="update_button"):
                st.rerun()
            
        else:
            st.info("🎥 Haz clic en 'START' para iniciar la webcam y comenzar el análisis")
    
    with col2:
        st.subheader("📊 Estadísticas")
        
        # Gráfico de emociones
        emotion_history = st.session_state.get('emotion_history', [])
        if emotion_history:
            fig = create_emotion_chart(emotion_history)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📈 Los gráficos aparecerán cuando se detecten emociones")
        
        # Controles
        st.subheader("💾 Acciones")
        
        # Botón para guardar captura
        if st.button("📸 Guardar Captura", type="primary"):
            if webrtc_ctx.state.playing:
                st.success("✅ Captura guardada (funcionalidad en desarrollo)")
        
        # Botón para resetear estadísticas
        if st.button("🔄 Resetear Estadísticas"):
            st.session_state.emotion_history = []
            st.session_state.current_emotion = 'neutral'
            st.session_state.current_confidence = 0.0
            st.session_state.current_age = 25
            st.session_state.current_gender = 'unknown'
            st.session_state.current_gender_confidence = 0.0
            st.session_state.analysis_count = 0
            transformer.emotion_history = []
            st.rerun()
        
        # Información adicional
        st.subheader("ℹ️ Información")
        st.info("""
        **Cómo usar:**
        1. Haz clic en 'START' para activar la cámara
        2. Permite acceso a la cámara cuando se solicite
        3. Mira a la cámara y cambia expresiones
        4. Observa los resultados en tiempo real
        5. Usa el botón "Actualizar Datos" si no se actualiza
        
        **Optimizado para reducir lag:**
        - Análisis cada 30 frames
        - Cooldown de 4 segundos entre análisis
        - Gestión mejorada de archivos temporales
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🧠 FaceMood AI - Análisis de emociones con DeepFace y Streamlit</p>
        <p>Desarrollado con ❤️ para demostrar el poder de la visión artificial</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 