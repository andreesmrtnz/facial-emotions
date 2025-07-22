#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceMood AI - Real Time - Detección de emociones en tiempo real continuo
=======================================================================

Versión con WebRTC para análisis en tiempo real continuo con gráficos.
"""

import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
import tempfile
import os
from deepface import DeepFace
import plotly.express as px
from collections import defaultdict
import threading
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="FaceMood AI - Real Time",
    page_icon="🧠",
    layout="wide"
)

# Configuración RTC para WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class FaceMoodTransformer(VideoTransformerBase):
    """Transformador de video para análisis en tiempo real continuo."""
    
    def __init__(self):
        self.frame_count = 0
        self.analysis_interval = 15  # Analizar cada 15 frames (más frecuente)
        self.current_results = {
            'emotion': 'neutral',
            'age': 25,
            'gender': 'unknown',
            'confidence': 0.0
        }
        self.emotion_history = []
        self.lock = threading.Lock()
        self.last_analysis_time = 0
        self.analysis_cooldown = 1.0  # 1 segundo entre análisis (más rápido)
        
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
        """Procesa cada frame del video en tiempo real."""
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
                    
                    # Extraer datos
                    emotions = analysis.get('emotion', {})
                    dominant_emotion = max(emotions, key=emotions.get) if emotions else 'neutral'
                    emotion_confidence = emotions.get(dominant_emotion, 0) / 100.0
                    age = analysis.get('age', 25)
                    gender = analysis.get('dominant_gender', 'unknown')
                    
                    # Actualizar resultados
                    with self.lock:
                        self.current_results = {
                            'emotion': dominant_emotion,
                            'confidence': emotion_confidence,
                            'age': age,
                            'gender': gender,
                            'all_emotions': emotions,
                            'timestamp': current_time
                        }
                        
                        # Agregar a historial con timestamp
                        self.emotion_history.append({
                            'time': current_time,
                            'emotion': dominant_emotion,
                            'confidence': emotion_confidence,
                            'age': age,
                            'gender': gender
                        })
                        
                        # Mantener solo los últimos 100 registros para gráficos
                        if len(self.emotion_history) > 100:
                            self.emotion_history.pop(0)
                
                # Limpiar archivo temporal
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"Error en análisis: {e}")
        
        self.frame_count += 1
        
        # Dibujar resultados en el frame
        img = self.draw_results_on_frame(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def draw_results_on_frame(self, img):
        """Dibuja los resultados en el frame."""
        with self.lock:
            results = self.current_results
        
        if not results:
            return img
        
        # Detectar rostros
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            emotion = results['emotion']
            color = self.colors.get(emotion, (255, 255, 255))
            emoji = self.emotion_emojis.get(emotion, '❓')
            
            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            # Agregar texto con emoción
            text = f"{emoji} {emotion.upper()}"
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Agregar edad y género
            age_text = f"Age: {results['age']}"
            gender_text = f"Gender: {results['gender']}"
            cv2.putText(img, age_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, gender_text, (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Agregar confianza
            conf_text = f"Conf: {results['confidence']:.1%}"
            cv2.putText(img, conf_text, (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img

def create_emotion_chart(emotion_history):
    """Crea un gráfico de las emociones detectadas en tiempo real."""
    if not emotion_history:
        return None
    
    # Contar emociones
    emotion_counts = defaultdict(int)
    for entry in emotion_history:
        emotion_counts[entry['emotion']] += 1
    
    if not emotion_counts:
        return None
    
    # Crear gráfico de barras
    fig = px.bar(
        x=list(emotion_counts.keys()),
        y=list(emotion_counts.values()),
        title="Emociones Detectadas en Tiempo Real",
        labels={'x': 'Emoción', 'y': 'Frecuencia'},
        color=list(emotion_counts.values()),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Emoción",
        yaxis_title="Frecuencia",
        showlegend=False,
        height=400
    )
    
    return fig

def create_timeline_chart(emotion_history):
    """Crea un gráfico de línea temporal de emociones."""
    if len(emotion_history) < 2:
        return None
    
    # Preparar datos para timeline
    df = pd.DataFrame(emotion_history)
    df['time_formatted'] = pd.to_datetime(df['time'], unit='s')
    
    # Crear gráfico de línea
    fig = px.line(
        df, 
        x='time_formatted', 
        y='confidence',
        color='emotion',
        title="Evolución de Emociones en Tiempo Real",
        labels={'time_formatted': 'Tiempo', 'confidence': 'Confianza', 'emotion': 'Emoción'}
    )
    
    fig.update_layout(
        xaxis_title="Tiempo",
        yaxis_title="Confianza",
        height=300
    )
    
    return fig

def main():
    """Función principal de la aplicación."""
    
    # Título y descripción
    st.title("🧠 FaceMood AI - Real Time")
    st.markdown("**Detección de emociones en tiempo real continuo con gráficos dinámicos**")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        **🎥 Cómo usar:**
        1. Haz clic en "START" para activar la cámara
        2. Permite acceso a tu cámara cuando el navegador lo solicite
        3. Mira a la cámara y cambia expresiones faciales
        4. Observa los resultados y gráficos en tiempo real
        
        **📊 Funcionalidades:**
        - Análisis continuo en tiempo real
        - Gráficos dinámicos actualizados
        - Detección de emociones, edad y género
        - Historial de análisis
        """)
        
        st.header("⚙️ Configuración")
        st.info("Esta versión usa WebRTC para análisis continuo en tiempo real.")
        
        # Controles
        st.header("🔧 Controles")
        if st.button("🔄 Limpiar Historial"):
            st.session_state.clear()
            st.rerun()
    
    # Inicializar session state
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    
    # Contenedor principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Cámara en Tiempo Real")
        
        # WebRTC Streamer
        webrtc_ctx = webrtc_streamer(
            key="facemood_realtime",
            video_transformer_factory=FaceMoodTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing:
            st.success("✅ Cámara activa - Analizando emociones en tiempo real...")
        else:
            st.info("👆 Haz clic en 'START' para activar la cámara y comenzar el análisis")
    
    with col2:
        st.header("📊 Resultados en Tiempo Real")
        
        if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
            transformer = webrtc_ctx.video_transformer
            
            with transformer.lock:
                results = transformer.current_results
                emotion_history = transformer.emotion_history
            
            # Mostrar resultados actuales
            if results and 'timestamp' in results:
                emotion = results['emotion']
                emoji = transformer.emotion_emojis.get(emotion, '❓')
                
                st.metric(
                    label="Emoción Detectada",
                    value=f"{emoji} {emotion.upper()}",
                    delta=f"{results['confidence']:.1%} confianza"
                )
                
                st.metric(
                    label="Edad Estimada",
                    value=f"{results['age']} años"
                )
                
                st.metric(
                    label="Género",
                    value=results['gender'].upper()
                )
                
                st.metric(
                    label="Análisis Realizados",
                    value=len(emotion_history)
                )
                
                # Información de tiempo
                if 'timestamp' in results:
                    time_diff = time.time() - results['timestamp']
                    st.info(f"⏱️ Último análisis: {time_diff:.1f}s atrás")
                
                # Mostrar todas las emociones
                if 'all_emotions' in results:
                    st.subheader("📈 Todas las Emociones")
                    emotions_df = pd.DataFrame(
                        list(results['all_emotions'].items()),
                        columns=['Emoción', 'Confianza']
                    )
                    emotions_df['Confianza'] = emotions_df['Confianza'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(emotions_df, use_container_width=True)
            else:
                st.info("👀 Esperando detección de rostro...")
        else:
            st.info("🎥 Activa la cámara para ver resultados")
    
    # Gráficos en tiempo real
    st.header("📈 Gráficos en Tiempo Real")
    
    if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
        transformer = webrtc_ctx.video_transformer
        
        with transformer.lock:
            emotion_history = transformer.emotion_history
        
        if emotion_history:
            # Gráfico de barras
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("📊 Frecuencia de Emociones")
                chart = create_emotion_chart(emotion_history)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            with col_chart2:
                st.subheader("⏰ Evolución Temporal")
                timeline_chart = create_timeline_chart(emotion_history)
                if timeline_chart:
                    st.plotly_chart(timeline_chart, use_container_width=True)
            
            # Tabla de historial reciente
            st.subheader("📋 Historial Reciente")
            if emotion_history:
                recent_data = emotion_history[-10:]  # Últimos 10 análisis
                df_recent = pd.DataFrame(recent_data)
                df_recent['time_formatted'] = pd.to_datetime(df_recent['time'], unit='s').dt.strftime('%H:%M:%S')
                df_recent['emotion_emoji'] = df_recent['emotion'].map(transformer.emotion_emojis)
                df_recent['emotion_display'] = df_recent['emotion_emoji'] + ' ' + df_recent['emotion'].str.upper()
                
                display_df = df_recent[['time_formatted', 'emotion_display', 'confidence', 'age', 'gender']].copy()
                display_df.columns = ['Hora', 'Emoción', 'Confianza', 'Edad', 'Género']
                display_df['Confianza'] = display_df['Confianza'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("📊 Los gráficos aparecerán cuando se detecten emociones")
    else:
        st.info("🎥 Activa la cámara para ver los gráficos en tiempo real")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🧠 FaceMood AI - Análisis en Tiempo Real Continuo</p>
        <p>💡 WebRTC + DeepFace + Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 