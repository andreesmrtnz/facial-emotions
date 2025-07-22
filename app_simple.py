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

# Configuración de la página
st.set_page_config(
    page_title="FaceMood AI - Simple",
    page_icon="🧠",
    layout="wide"
)

def analyze_frame(frame):
    """Analiza un frame y devuelve los resultados."""
    try:
        # Crear archivo temporal
        temp_filename = f"temp_frame_{int(time.time() * 1000)}.jpg"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Guardar frame
        cv2.imwrite(temp_path, frame)
        
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
            
            # Limpiar archivo temporal
            try:
                os.remove(temp_path)
            except:
                pass
            
            return {
                'emotion': dominant_emotion,
                'confidence': emotion_confidence,
                'age': age,
                'gender': gender,
                'all_emotions': emotions
            }
    except Exception as e:
        print(f"Error en análisis: {e}")
        
    return None

def draw_results_on_frame(frame, results):
    """Dibuja los resultados en el frame."""
    if not results:
        return frame
    
    # Detectar rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Mapeo de emociones a emojis y colores
    emotion_emojis = {
        'angry': '😠', 'disgust': '🤢', 'fear': '😨',
        'happy': '😀', 'sad': '😢', 'surprise': '😮', 'neutral': '😐'
    }
    
    colors = {
        'angry': (0, 0, 255), 'disgust': (0, 255, 0), 'fear': (255, 0, 255),
        'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (255, 165, 0),
        'neutral': (128, 128, 128)
    }
    
    for (x, y, w, h) in faces:
        emotion = results['emotion']
        color = colors.get(emotion, (255, 255, 255))
        emoji = emotion_emojis.get(emotion, '❓')
        
        # Dibujar rectángulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Preparar texto
        emotion_text = f"{emoji} {emotion.upper()}"
        age_gender_text = f"Age: {results['age']} | {results['gender'].title()}"
        confidence_text = f"Conf: {results['confidence']:.2f}"
        
        # Configurar fuente
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Obtener tamaños de texto
        (emotion_width, emotion_height), _ = cv2.getTextSize(emotion_text, font, font_scale, thickness)
        (age_width, age_height), _ = cv2.getTextSize(age_gender_text, font, 0.5, 1)
        
        # Dibujar fondo
        cv2.rectangle(frame, (x, y - emotion_height - 80), (x + max(emotion_width, age_width) + 15, y), color, -1)
        
        # Dibujar textos
        cv2.putText(frame, emotion_text, (x + 5, y - 55), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, age_gender_text, (x + 5, y - 35), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, confidence_text, (x + 5, y - 15), font, 0.5, (255, 255, 255), 1)
    
    return frame

def create_emotion_chart(emotion_history):
    """Crea gráfico de emociones."""
    if not emotion_history:
        return None
    
    emotion_counts = defaultdict(int)
    for emotion in emotion_history:
        emotion_counts[emotion] += 1
    
    fig = px.bar(
        x=list(emotion_counts.keys()),
        y=list(emotion_counts.values()),
        title="Evolución de Emociones",
        labels={'x': 'Emociones', 'y': 'Frecuencia'},
        color=list(emotion_counts.keys()),
        color_discrete_map={
            'angry': '#FF0000', 'disgust': '#00FF00', 'fear': '#FF00FF',
            'happy': '#FFFF00', 'sad': '#0000FF', 'surprise': '#FFA500',
            'neutral': '#808080'
        }
    )
    
    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def main():
    st.title("🧠 FaceMood AI - Simple")
    st.markdown("### Detección de emociones en tiempo real - Versión optimizada")
    
    # Inicializar session state
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    # Sidebar con configuración
    st.sidebar.title("⚙️ Configuración")
    analysis_interval = st.sidebar.slider(
        "Frecuencia de análisis (frames)", 
        min_value=10, 
        max_value=50, 
        value=15,
        help="Menor valor = más frecuente (pero más lento)"
    )
    
    cooldown_time = st.sidebar.slider(
        "Tiempo entre análisis (segundos)", 
        min_value=1.0, 
        max_value=5.0, 
        value=1.5,
        step=0.5,
        help="Menor valor = más análisis por minuto"
    )
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Cámara Web")
        
        # Placeholder para la imagen
        image_placeholder = st.empty()
        
        # Controles
        start_button = st.button("🎥 Iniciar Cámara", type="primary")
        stop_button = st.button("⏹️ Detener")
        
        if start_button:
            st.session_state.running = True
        
        if stop_button:
            st.session_state.running = False
        
        # Mostrar estado
        if st.session_state.running:
            st.success("✅ Cámara activa - Analizando emociones...")
        else:
            st.info("📱 Presiona 'Iniciar Cámara' para comenzar")
        
        # Captura de video
        if st.session_state.running:
            cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                # Configurar análisis
                frame_count = 0
                last_analysis_time = 0
                current_results = None
                
                # Loop principal
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Voltear frame
                    frame = cv2.flip(frame, 1)
                    
                    # Analizar cada cierto número de frames (MEJORADO)
                    current_time = time.time()
                    if (frame_count % analysis_interval == 0 and 
                        current_time - last_analysis_time > cooldown_time):
                        
                        # Analizar frame
                        results = analyze_frame(frame)
                        if results:
                            current_results = results
                            st.session_state.last_analysis = results
                            st.session_state.analysis_count += 1
                            st.session_state.emotion_history.append(results['emotion'])
                            
                            # Mantener solo últimos 30 (aumentado)
                            if len(st.session_state.emotion_history) > 30:
                                st.session_state.emotion_history.pop(0)
                            
                            last_analysis_time = current_time
                            print(f"🎭 Emoción: {results['emotion']} ({results['confidence']:.2f}) - Edad: {results['age']} - Género: {results['gender']}")
                    
                    # Dibujar resultados en frame
                    if current_results:
                        frame = draw_results_on_frame(frame, current_results)
                    
                    # Convertir BGR a RGB para Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Mostrar frame (CORREGIDO: use_container_width)
                    image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    frame_count += 1
                    
                    # Control de FPS mejorado
                    time.sleep(0.025)  # ~40 FPS
                
                cap.release()
            else:
                st.error("❌ No se pudo acceder a la cámara")
                st.session_state.running = False
    
    with col2:
        st.subheader("📊 Resultados en Tiempo Real")
        
        # Mostrar último análisis
        if st.session_state.last_analysis:
            results = st.session_state.last_analysis
            
            # Mapeo de emojis
            emotion_emojis = {
                'angry': '😠', 'disgust': '🤢', 'fear': '😨',
                'happy': '😀', 'sad': '😢', 'surprise': '😮', 'neutral': '😐'
            }
            
            # Métricas en columnas
            col_met1, col_met2 = st.columns(2)
            
            with col_met1:
                st.metric("Emoción", 
                         f"{emotion_emojis.get(results['emotion'], '❓')} {results['emotion'].title()}")
                st.metric("Confianza", f"{results['confidence']:.1%}")
            
            with col_met2:
                st.metric("Edad", f"{results['age']} años")
                st.metric("Género", results['gender'].title())
            
            st.metric("Análisis Realizados", st.session_state.analysis_count)
            
            # Información detallada
            st.info(f"🎭 **Última emoción**: {emotion_emojis.get(results['emotion'], '❓')} {results['emotion'].title()} ({results['confidence']:.1%})")
        
        # Gráfico de emociones
        st.subheader("📈 Estadísticas")
        
        if st.session_state.emotion_history:
            fig = create_emotion_chart(st.session_state.emotion_history)
            if fig:
                # CORREGIDO: use_container_width en lugar de use_column_width
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Los gráficos aparecerán cuando se detecten emociones")
        
        # Controles
        st.subheader("🔧 Controles")
        
        if st.button("🔄 Resetear Estadísticas"):
            st.session_state.emotion_history = []
            st.session_state.last_analysis = None
            st.session_state.analysis_count = 0
            st.rerun()
        
        # Información
        st.subheader("ℹ️ Información")
        st.info(f"""
        **Configuración actual:**
        - Análisis cada {analysis_interval} frames
        - Cooldown: {cooldown_time}s entre análisis
        - Cámara a ~40 FPS
        - Actualización en tiempo real
        """)

if __name__ == "__main__":
    main() 