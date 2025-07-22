# 🧠 Sistema de Reconocimiento de Emociones en Tiempo Real

## 📋 **Descripción del Proyecto**

Este proyecto incluye **dos versiones** de un sistema de reconocimiento de emociones:

### 🖥️ **Versión Desktop** (`emotion_detection.py`)
- Aplicación de escritorio con OpenCV
- Detección de emociones en tiempo real
- Gráficos de evolución emocional
- Almacenamiento de sesiones

### 🌐 **Versión Web** (`app.py`) - **FaceMood AI**
- Aplicación web con Streamlit
- Análisis de emociones, edad y género
- Interfaz interactiva moderna
- Lista para desplegar en Hugging Face Spaces

---

## 🚀 **FaceMood AI - Aplicación Web**

**Descubre tu emoción, edad y género en tiempo real con inteligencia artificial**

### ✨ **Características Principales**

- 🎭 **Detección de 7 emociones**: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- 👤 **Análisis de género**: Identificación del género de la persona
- 📅 **Estimación de edad**: Edad aproximada con alta precisión
- 📊 **Gráficos en tiempo real**: Evolución de emociones con Plotly
- 💾 **Guardado de capturas**: Conserva momentos especiales
- 🎨 **Interfaz moderna**: Diseño responsive y atractivo

### 🛠️ **Tecnologías Utilizadas**

- **DeepFace**: Análisis de emociones, edad y género
- **Streamlit**: Framework web interactivo
- **Streamlit-WebRTC**: Integración de webcam en tiempo real
- **Plotly**: Gráficos dinámicos
- **OpenCV**: Procesamiento de imágenes
- **TensorFlow**: Framework de deep learning

### 🎮 **Cómo Usar la Aplicación Web**

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements_web.txt
   ```

2. **Ejecutar la aplicación**:
   ```bash
   streamlit run app.py
   ```

3. **Abrir en el navegador**:
   ```
   http://localhost:8501
   ```

4. **Permitir acceso a la cámara** y disfrutar del análisis en tiempo real

### 🌍 **Despliegue en Hugging Face Spaces**

La aplicación está lista para desplegar en Hugging Face Spaces:

1. Crear un nuevo Space en Hugging Face
2. Subir los archivos: `app.py`, `requirements_web.txt`, `README.md`
3. Configurar como Streamlit App
4. ¡Listo! Tu app estará disponible públicamente

---

## 🖥️ **Versión Desktop - Instrucciones**

### 📋 **Requisitos Previos**

1. **Python 3.11** instalado en tu sistema (compatible con DeepFace)
2. **Cámara web** funcional
3. **Conexión a internet** (solo para instalar dependencias)

### 🔧 **Instalación de Dependencias**

```bash
pip install -r requirements.txt
```

### 🎯 **Ejecución del Programa**

```bash
python emotion_detection.py
```

### 🎮 **Controles**

- **Presiona 'q'** para salir del programa
- **Presiona 'g'** para mostrar/ocultar gráfico de evolución emocional
- **Presiona 's'** para activar/desactivar guardado de sesión

### 🎭 **Emociones Detectadas**

- 😀 **Happy** (Feliz) - Amarillo
- 😢 **Sad** (Triste) - Azul
- 😠 **Angry** (Enfadado) - Rojo
- 😮 **Surprise** (Sorprendido) - Naranja
- 😐 **Neutral** (Neutral) - Gris
- 😨 **Fear** (Miedo) - Magenta
- 🤢 **Disgust** (Asco) - Verde

---

## 📊 **Comparación de Versiones**

| Característica | Versión Desktop | Versión Web (FaceMood AI) |
|---|---|---|
| **Plataforma** | Escritorio | Web (navegador) |
| **Detección de emociones** | ✅ | ✅ |
| **Detección de edad** | ❌ | ✅ |
| **Detección de género** | ❌ | ✅ |
| **Gráficos interactivos** | Básicos | Avanzados (Plotly) |
| **Guardado de capturas** | JSON | JSON + Imagen |
| **Acceso remoto** | ❌ | ✅ |
| **Interfaz** | OpenCV | Streamlit |
| **Despliegue** | Local | Hugging Face Spaces |

---

## 🏆 **Resultados y Métricas**

### ✅ **Funcionalidades Implementadas**
- **Detección real de emociones** usando DeepFace
- **7 emociones detectables** con alta precisión
- **Gráfico de evolución emocional** en tiempo real
- **Almacenamiento de sesiones** con timestamps
- **Interfaz visual profesional** con emojis y colores
- **Aplicación web completa** lista para producción

### 🎯 **Objetivos Cumplidos**
- ✅ Detectar rostros humanos en tiempo real
- ✅ Clasificar emociones usando modelo pre-entrenado
- ✅ Visualizar emociones sobre el rostro con emojis
- ✅ Mostrar gráfico de evolución emocional
- ✅ Crear interfaz funcional y atractiva
- ✅ Documentar el proyecto completamente
- ✅ Crear aplicación web desplegable

---

## 📁 **Estructura del Proyecto**

```
facial-emotions/
├── emotion_detection.py          # Versión desktop
├── app.py                        # Versión web (FaceMood AI)
├── requirements.txt              # Dependencias desktop
├── requirements_web.txt          # Dependencias web
├── README.md                     # Documentación principal
├── README_web.md                 # Documentación web
├── INSTRUCCIONES.md              # Instrucciones detalladas
└── .gitattributes               # Configuración Git
```

---

## 🚀 **Próximos Pasos**

1. **Desplegar FaceMood AI** en Hugging Face Spaces
2. **Compartir en LinkedIn** con capturas de pantalla
3. **Crear GIF animado** de la aplicación funcionando
4. **Añadir funcionalidades avanzadas**:
   - Detección de múltiples personas
   - Análisis de micro-expresiones
   - Predicción de personalidad
   - Modo educativo explicativo

---

<div align="center">

**¿Te gustó el proyecto? ¡Dale una ⭐ al repositorio!**

*Desarrollado con ❤️ para demostrar el poder de la visión artificial*

</div>