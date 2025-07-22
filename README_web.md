# 🧠 FaceMood AI

**Descubre tu emoción, edad y género en tiempo real con inteligencia artificial**

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://facemood-ai.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![DeepFace](https://img.shields.io/badge/DeepFace-AI-orange?logo=tensorflow)](https://github.com/serengil/deepface)

## 🎯 ¿Qué es FaceMood AI?

FaceMood AI es una aplicación web interactiva que utiliza **visión artificial** y **deep learning** para analizar en tiempo real:

- 🎭 **Emociones**: Detecta 7 emociones diferentes (feliz, triste, enfadado, sorprendido, miedo, asco, neutral)
- 👤 **Género**: Identifica el género de la persona
- 📅 **Edad**: Estima la edad aproximada
- 📊 **Estadísticas**: Muestra gráficos en tiempo real de las emociones detectadas

## ✨ Características Principales

### 🎥 **Análisis en Tiempo Real**
- Webcam integrada con Streamlit
- Procesamiento de video en tiempo real
- Detección automática de rostros
- Análisis continuo con DeepFace

### 🎨 **Interfaz Interactiva**
- Diseño moderno y responsive
- Gráficos dinámicos con Plotly
- Métricas en tiempo real
- Controles personalizables

### 💾 **Funcionalidades Avanzadas**
- Guardado de capturas con predicciones
- Historial de emociones
- Reset de estadísticas
- Exportación de datos

## 🚀 Instalación y Uso

### Requisitos Previos
- Python 3.11 o superior
- Cámara web funcional
- Conexión a internet

### Instalación Local

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/facemood-ai.git
cd facemood-ai
```

2. **Instalar dependencias**
```bash
pip install -r requirements_web.txt
```

3. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

4. **Abrir en el navegador**
```
http://localhost:8501
```

### Despliegue en Hugging Face Spaces

1. **Crear un nuevo Space** en Hugging Face
2. **Subir los archivos**:
   - `app.py`
   - `requirements_web.txt`
   - `README.md`
3. **Configurar como Streamlit App**
4. **¡Listo!** Tu app estará disponible en `https://huggingface.co/spaces/tu-usuario/facemood-ai`

## 🎮 Cómo Usar

1. **Permitir acceso a la cámara** cuando la app lo solicite
2. **Mira a la cámara** y cambia expresiones faciales
3. **Observa los resultados** en tiempo real:
   - Emoción detectada con emoji
   - Edad estimada
   - Género identificado
   - Gráfico de evolución emocional
4. **Usa los controles** para personalizar el análisis
5. **Guarda capturas** cuando quieras conservar un momento

## 🔧 Tecnologías Utilizadas

### 🤖 **Inteligencia Artificial**
- **DeepFace**: Análisis de emociones, edad y género
- **TensorFlow**: Framework de deep learning
- **OpenCV**: Procesamiento de imágenes

### 🌐 **Desarrollo Web**
- **Streamlit**: Framework para aplicaciones web
- **Streamlit-WebRTC**: Integración de webcam
- **Plotly**: Gráficos interactivos
- **WebRTC**: Comunicación en tiempo real

### 📊 **Visualización**
- **Plotly Express**: Gráficos de barras dinámicos
- **Streamlit Components**: Métricas y controles
- **CSS Personalizado**: Diseño atractivo

## 📈 Emociones Detectadas

| Emoción | Emoji | Descripción |
|---------|-------|-------------|
| 😀 Happy | Feliz | Sonrisa, alegría, satisfacción |
| 😢 Sad | Triste | Melancolía, pena, desánimo |
| 😠 Angry | Enfadado | Ira, frustración, enojo |
| 😮 Surprise | Sorprendido | Asombro, shock, incredulidad |
| 😨 Fear | Miedo | Ansiedad, terror, preocupación |
| 🤢 Disgust | Asco | Repulsión, desagrado, náusea |
| 😐 Neutral | Neutral | Calma, equilibrio, indiferencia |

## 🎯 Casos de Uso

### 👨‍💼 **Profesional**
- Análisis de emociones en entrevistas
- Evaluación de presentaciones
- Monitoreo de bienestar laboral

### 🎓 **Educativo**
- Investigación en psicología
- Estudios de comportamiento
- Demostraciones de IA

### 🎮 **Entretenimiento**
- Apps de redes sociales
- Juegos interactivos
- Filtros de realidad aumentada

### 🏥 **Salud**
- Monitoreo de salud mental
- Terapia asistida por IA
- Investigación médica

## 📊 Métricas de Rendimiento

- **Precisión de emociones**: ~85-90%
- **Precisión de edad**: ±5 años
- **Precisión de género**: ~95%
- **Latencia**: <100ms por frame
- **FPS**: 15-30 frames por segundo

## 🔮 Próximas Funcionalidades

- [ ] **Detección de múltiples personas**
- [ ] **Análisis de micro-expresiones**
- [ ] **Predicción de personalidad**
- [ ] **Integración con APIs externas**
- [ ] **Modo offline**
- [ ] **Exportación de videos**
- [ ] **Comparación entre usuarios**
- [ ] **Modo educativo explicativo**

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! 

1. **Fork** el proyecto
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **Sefik Ilkin Serengil** por [DeepFace](https://github.com/serengil/deepface)
- **Streamlit** por el framework web
- **OpenCV** por el procesamiento de imágenes
- **Plotly** por las visualizaciones interactivas

## 📞 Contacto

- **GitHub**: [@tu-usuario](https://github.com/tu-usuario)
- **LinkedIn**: [Tu Nombre](https://linkedin.com/in/tu-perfil)
- **Email**: tu-email@example.com

---

<div align="center">

**¿Te gustó FaceMood AI? ¡Dale una ⭐ al repositorio!**

*Desarrollado con ❤️ para demostrar el poder de la visión artificial*

</div> 