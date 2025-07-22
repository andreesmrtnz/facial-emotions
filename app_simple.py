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
        
        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Agregar texto con emoción
        text = f"{emoji} {emotion.upper()}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Agregar edad y género
        age_text = f"Age: {results['age']}"
        gender_text = f"Gender: {results['gender']}"
        cv2.putText(frame, age_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, gender_text, (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def create_emotion_chart(emotion_history):
    """Crea un gráfico de las emociones detectadas."""
    if not emotion_history:
        return None
    
    # Contar emociones
    emotion_counts = defaultdict(int)
    for emotion in emotion_history:
        emotion_counts[emotion] += 1
    
    if not emotion_counts:
        return None
    
    # Crear gráfico
    fig = px.bar(
        x=list(emotion_counts.keys()),
        y=list(emotion_counts.values()),
        title="Emociones Detectadas",
        labels={'x': 'Emoción', 'y': 'Frecuencia'},
        color=list(emotion_counts.values()),
        color_continuous_scale='viridis'
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
    if 'last_photo_time' not in st.session_state:
        st.session_state.last_photo_time = 0
    
    # Sidebar con configuración
    st.sidebar.title("⚙️ Configuración")
    
    # Configuración de análisis automático
    auto_analyze = st.sidebar.checkbox(
        "🔄 Análisis Automático", 
        value=True,
        help="Analiza automáticamente cada foto capturada"
    )
    
    analysis_interval = st.sidebar.slider(
        "⏱️ Intervalo de análisis (segundos)", 
        min_value=1, 
        max_value=10, 
        value=3,
        help="Tiempo mínimo entre análisis automáticos"
    )
    
    # Información sobre la cámara
    st.sidebar.info("""
    **📹 Cómo usar:**
    1. Activa "Análisis Automático"
    2. Haz clic en "Take photo" para capturar
    3. El análisis se ejecuta automáticamente
    4. Los resultados se muestran en tiempo real
    5. Los gráficos se actualizan automáticamente
    """)
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Cámara Web")
        
        # Usar st.camera_input para captura de fotos
        camera_photo = st.camera_input(
            label="Haz clic en 'Take photo' para capturar y analizar automáticamente",
            help="Captura una foto para analizar emociones, edad y género"
        )
        
        # Analizar foto automáticamente cuando se capture
        if camera_photo is not None:
            current_time = time.time()
            
            # Verificar si ha pasado suficiente tiempo desde el último análisis
            if (auto_analyze and 
                current_time - st.session_state.last_photo_time > analysis_interval):
                
                st.session_state.last_photo_time = current_time
                
                # Convertir la imagen de Streamlit a formato OpenCV
                bytes_data = camera_photo.getvalue()
                
                # Convertir bytes a numpy array
                nparr = np.frombuffer(bytes_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Mostrar que está analizando
                with st.spinner("🔍 Analizando imagen..."):
                    # Analizar frame
                    results = analyze_frame(frame)
                
                if results:
                    st.session_state.last_analysis = results
                    st.session_state.analysis_count += 1
                    st.session_state.emotion_history.append(results['emotion'])
                    
                    # Mantener solo últimos 50
                    if len(st.session_state.emotion_history) > 50:
                        st.session_state.emotion_history.pop(0)
                    
                    # Dibujar resultados en frame
                    frame_with_results = draw_results_on_frame(frame.copy(), results)
                    
                    # Convertir BGR a RGB para mostrar
                    frame_rgb = cv2.cvtColor(frame_with_results, cv2.COLOR_BGR2RGB)
                    
                    # Mostrar imagen con resultados
                    st.image(frame_rgb, caption="Imagen analizada con resultados", use_container_width=True)
                    
                    # Mostrar resultado
                    emotion_emojis = {
                        'angry': '😠', 'disgust': '🤢', 'fear': '😨',
                        'happy': '😀', 'sad': '😢', 'surprise': '😮', 'neutral': '😐'
                    }
                    emoji = emotion_emojis.get(results['emotion'], '❓')
                    
                    st.success(f"✅ Análisis completado: {emoji} {results['emotion'].title()} ({results['confidence']:.1%}) - Edad: {results['age']} - Género: {results['gender']}")
                    
                    # Auto-rerun para actualizar gráficos
                    st.rerun()
                else:
                    st.warning("⚠️ No se pudo detectar un rostro en la imagen")
            elif not auto_analyze:
                st.info("📸 Foto capturada. Activa 'Análisis Automático' para procesar.")
            else:
                st.info(f"⏳ Esperando {analysis_interval - (current_time - st.session_state.last_photo_time):.1f}s para el próximo análisis...")
        else:
            st.info("📱 Haz clic en 'Take photo' para comenzar el análisis")
    
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
            
            # Mostrar todas las emociones si están disponibles
            if 'all_emotions' in results:
                st.subheader("📈 Todas las Emociones")
                emotions_data = []
                for emotion, confidence in results['all_emotions'].items():
                    emotions_data.append({
                        'Emoción': emotion.title(),
                        'Confianza': f"{confidence:.1f}%"
                    })
                st.dataframe(emotions_data, use_container_width=True)
        
        # Gráfico de emociones
        st.subheader("📈 Estadísticas")
        
        if st.session_state.emotion_history:
            fig = create_emotion_chart(st.session_state.emotion_history)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Los gráficos aparecerán cuando se detecten emociones")
        
        # Controles
        st.subheader("🔧 Controles")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔄 Resetear Estadísticas"):
                st.session_state.emotion_history = []
                st.session_state.last_analysis = None
                st.session_state.analysis_count = 0
                st.rerun()
        
        with col_btn2:
            if st.button("📊 Actualizar Gráficos"):
                st.rerun()
        
        # Información
        st.subheader("ℹ️ Información")
        st.info(f"""
        **Estado actual:**
        - Análisis realizados: {st.session_state.analysis_count}
        - Emociones en historial: {len(st.session_state.emotion_history)}
        - Análisis automático: {'✅ Activado' if auto_analyze else '❌ Desactivado'}
        - Intervalo: {analysis_interval}s
        - Última actualización: {datetime.now().strftime('%H:%M:%S')}
        """)

if __name__ == "__main__":
    main() 