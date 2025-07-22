# 🚀 Guía de Despliegue - FaceMood AI

## 📋 Opciones de Despliegue

### 1. **Streamlit Cloud (Recomendado) ⭐**

#### Pasos:
1. **Crear cuenta** en [share.streamlit.io](https://share.streamlit.io)
2. **Conectar tu repositorio** de GitHub
3. **Configurar la aplicación**:
   - **Main file path**: `app.py`
   - **Python version**: 3.11
4. **Hacer deploy** - ¡Automático!

#### Ventajas:
- ✅ Gratuito
- ✅ Despliegue automático
- ✅ Integración con GitHub
- ✅ SSL automático
- ✅ Escalabilidad

---

### 2. **Hugging Face Spaces**

#### Pasos:
1. **Crear un nuevo Space** en [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Seleccionar Streamlit** como SDK
3. **Subir archivos**:
   - `app.py`
   - `requirements_web.txt`
   - `packages.txt`
   - `.streamlit/config.toml`
4. **Configurar** como Streamlit App

#### Ventajas:
- ✅ Gratuito
- ✅ Comunidad de IA
- ✅ Integración con modelos

---

### 3. **Heroku**

#### Pasos:
1. **Instalar Heroku CLI**
2. **Crear `Procfile`**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
3. **Configurar variables de entorno**:
   ```bash
   heroku config:set OPENCV_VIDEOIO_PRIORITY_MSMF=0
   ```
4. **Desplegar**:
   ```bash
   heroku create facemood-ai
   git push heroku main
   ```

---

### 4. **Google Cloud Run**

#### Pasos:
1. **Crear `Dockerfile`**:
   ```dockerfile
   FROM python:3.11-slim
   COPY requirements_web.txt .
   RUN pip install -r requirements_web.txt
   COPY . .
   EXPOSE 8080
   CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
   ```
2. **Construir y desplegar**:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/facemood-ai
   gcloud run deploy facemood-ai --image gcr.io/PROJECT_ID/facemood-ai --platform managed
   ```

---

## 🔧 Configuración Requerida

### Archivos Necesarios:
- ✅ `app.py` - Aplicación principal
- ✅ `requirements_web.txt` - Dependencias Python
- ✅ `packages.txt` - Dependencias del sistema
- ✅ `.streamlit/config.toml` - Configuración de Streamlit

### Variables de Entorno (Opcional):
```bash
OPENCV_VIDEOIO_PRIORITY_MSMF=0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
```

---

## 🌐 URLs de Ejemplo

Una vez desplegado, tu aplicación estará disponible en:
- **Streamlit Cloud**: `https://tu-app.streamlit.app`
- **Hugging Face**: `https://huggingface.co/spaces/tu-usuario/facemood-ai`
- **Heroku**: `https://facemood-ai.herokuapp.com`

---

## 🐛 Solución de Problemas

### Error: "No module named 'cv2'"
- **Solución**: Usar `opencv-python-headless` en lugar de `opencv-python`

### Error: "WebRTC not supported"
- **Solución**: Verificar que `streamlit-webrtc` esté instalado correctamente

### Error: "Camera access denied"
- **Solución**: Configurar HTTPS (requerido para acceso a cámara)

### Error: "Memory limit exceeded"
- **Solución**: Optimizar el código o usar un plan con más memoria

---

## 📊 Monitoreo

### Métricas a Revisar:
- **Tiempo de carga** de la aplicación
- **Uso de memoria** durante la detección
- **Latencia** de respuesta
- **Errores** en consola

### Herramientas de Monitoreo:
- **Streamlit Cloud**: Dashboard integrado
- **Heroku**: Heroku Metrics
- **Google Cloud**: Cloud Monitoring

---

## 🔄 Actualizaciones

### Despliegue Automático:
1. **Hacer cambios** en tu código
2. **Commit y push** a GitHub
3. **Despliegue automático** (Streamlit Cloud/Hugging Face)

### Despliegue Manual:
```bash
git add .
git commit -m "Actualizar aplicación"
git push origin main
```

---

## 💡 Consejos de Optimización

1. **Usar `opencv-python-headless`** para servidores sin GUI
2. **Configurar timeouts** apropiados para WebRTC
3. **Optimizar modelos** de DeepFace para producción
4. **Implementar caché** para mejorar rendimiento
5. **Usar CDN** para archivos estáticos

---

## 🆘 Soporte

Si tienes problemas con el despliegue:
1. **Revisar logs** de la plataforma
2. **Verificar dependencias** en `requirements_web.txt`
3. **Probar localmente** antes de desplegar
4. **Consultar documentación** de la plataforma elegida 