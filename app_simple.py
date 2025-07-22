#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceMood AI - Simple - Detección de emociones en tiempo real
===========================================================

Versión simplificada que funciona realmente en tiempo real.
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

# Configuración de la página
st.set_page_config(
    page_title="FaceMood AI - Simple",
    page_icon="🧠",
    layout="wide"
)

# Configuración RTC para WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class FaceMoodTransformer(VideoTransformerBase):
    """Transformador de video para análisis en tiempo real con WebRTC."""
    
    def __init__(self):
        self.frame_count = 0
        self.analysis_interval = 30  # Analizar cada 30 frames
        self.current_results = {
            'emotion': 'neutral',
            'age': 25,
            'gender': 'unknown',
            'confidence': 0.0
        }
        self.emotion_history = []
        self.lock = threading.Lock()
        self.last_analysis_time = 0
        self.analysis_cooldown = 2.0  # 2 segundos entre análisis
        
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
        """Procesa cada frame del video."""
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
                            'all_emotions': emotions
                        }
                        
                        # Agregar a historial
                        self.emotion_history.append({
                            'time': current_time,
                            'emotion': dominant_emotion,
                            'confidence': emotion_confidence
                        })
                        
                        # Mantener solo los últimos 50 registros
                        if len(self.emotion_history) > 50:
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
        
        return img

def create_emotion_chart(emotion_history):
    """Crea un gráfico de las emociones detectadas."""
    if not emotion_history:
        return None
    
    # Contar emociones
    emotion_counts = defaultdict(int)
    for entry in emotion_history:
        emotion_counts[entry['emotion']] += 1
    
    if not emotion_counts:
        return None
    
    # Crear gráfico
    fig = px.bar(
        x=list(emotion_counts.keys()),
        y=list(emotion_counts.values()),
        title="Emociones Detectadas en Tiempo Real",
        labels={'x': 'Emoción', 'y': 'Frecuencia'},
        color=list(emotion_counts.values()),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def display_statistics(results, emotion_history, data_source, is_video_active):
    """Muestra las estadísticas en el lateral derecho."""
    st.subheader("📊 Resultados en Tiempo Real")
    
    # Mostrar fuente de datos
    if data_source:
        st.markdown(f"**{data_source}**")
    
    if results:
        emotion = results['emotion']
        emoji = '😀' if emotion == 'happy' else '😢' if emotion == 'sad' else '😠' if emotion == 'angry' else '😮' if emotion == 'surprise' else '😨' if emotion == 'fear' else '🤢' if emotion == 'disgust' else '😐'
        
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
        
        # Mostrar todas las emociones
        if 'all_emotions' in results:
            st.subheader("📈 Todas las Emociones")
            emotions_df = []
            for emotion_name, confidence in results['all_emotions'].items():
                emotions_df.append({
                    'Emoción': emotion_name.title(),
                    'Confianza': f"{confidence:.1f}%"
                })
            st.dataframe(emotions_df, use_container_width=True)
    else:
        if is_video_active:
            st.info("👀 Esperando detección de rostro...")
        else:
            st.info("🎥 No hay datos guardados. Activa el video para comenzar.")
    
    # Mostrar gráficos
    if emotion_history:
        st.subheader(f"📊 Historial de Emociones ({data_source})")
        chart = create_emotion_chart(emotion_history)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    else:
        if is_video_active:
            st.info("📊 Los gráficos aparecerán cuando se detecten emociones")
        else:
            st.info("📊 No hay historial guardado. Activa el video para generar datos.")

def main():
    st.title("🧠 FaceMood AI - Video en Tiempo Real")
    st.markdown("### Detección de emociones en video continuo - Análisis automático")
    
    # Inicializar session state para persistencia de datos
    if 'saved_results' not in st.session_state:
        st.session_state.saved_results = None
    if 'saved_emotion_history' not in st.session_state:
        st.session_state.saved_emotion_history = []
    if 'video_active' not in st.session_state:
        st.session_state.video_active = False
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    
    # Sidebar con configuración
    st.sidebar.title("⚙️ Configuración")
    
    # Información sobre el video en tiempo real
    st.sidebar.info("""
    **📹 Video en Tiempo Real:**
    1. Haz clic en "START" para activar la cámara
    2. Permite acceso a tu cámara cuando lo solicite
    3. El análisis se ejecuta automáticamente cada 2 segundos
    4. Los resultados se muestran en tiempo real
    5. Al dar "STOP" las estadísticas se guardan
    6. Los gráficos permanecen hasta el siguiente "START"
    """)
    
    # Botón para limpiar estadísticas guardadas
    if st.sidebar.button("🗑️ Limpiar Estadísticas Guardadas"):
        st.session_state.saved_results = None
        st.session_state.saved_emotion_history = []
        st.session_state.video_active = False
        st.rerun()
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video en Tiempo Real")
        
        # WebRTC Streamer para video continuo
        webrtc_ctx = webrtc_streamer(
            key="facemood_realtime",
            video_transformer_factory=FaceMoodTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Detectar cambios en el estado del video
        current_video_state = webrtc_ctx.state.playing
        
        if current_video_state and not st.session_state.video_active:
            # Video se acaba de activar
            st.session_state.video_active = True
            st.success("✅ Video activo - Analizando emociones en tiempo real...")
        elif not current_video_state and st.session_state.video_active:
            # Video se acaba de detener - guardar estadísticas
            st.session_state.video_active = False
            if webrtc_ctx.video_transformer:
                with webrtc_ctx.video_transformer.lock:
                    if webrtc_ctx.video_transformer.current_results:
                        st.session_state.saved_results = webrtc_ctx.video_transformer.current_results.copy()
                    if webrtc_ctx.video_transformer.emotion_history:
                        st.session_state.saved_emotion_history = webrtc_ctx.video_transformer.emotion_history.copy()
            st.info("⏸️ Video detenido - Estadísticas guardadas")
        elif current_video_state:
            st.success("✅ Video activo - Analizando emociones en tiempo real...")
        else:
            st.info("👆 Haz clic en 'START' para activar el video en tiempo real")
    
    with col2:
        # Determinar qué datos mostrar
        if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
            # Video activo - usar datos en tiempo real
            transformer = webrtc_ctx.video_transformer
            with transformer.lock:
                results = transformer.current_results
                emotion_history = transformer.emotion_history
            data_source = "🔄 EN VIVO"
            is_video_active = True
        else:
            # Video detenido - usar datos guardados
            results = st.session_state.saved_results
            emotion_history = st.session_state.saved_emotion_history
            data_source = "💾 GUARDADO" if results else None
            is_video_active = False
        
        # Mostrar estadísticas SIEMPRE en el lateral
        display_statistics(results, emotion_history, data_source, is_video_active)
    
    # Auto-refresh para actualizar estadísticas en tiempo real (solo cuando video activo)
    if webrtc_ctx.state.playing:
        # Refrescar cada 3 segundos cuando el video está activo
        time.sleep(3)
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🧠 FaceMood AI - Análisis de Video en Tiempo Real</p>
        <p>💡 Detecta emociones automáticamente mientras cambias expresiones</p>
        <p>📊 Estadísticas actualizadas automáticamente cada 3 segundos</p>
        <p>💾 Los datos se guardan al detener el video</p>
        <p>📱 Estadísticas siempre visibles en el lateral derecho</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 