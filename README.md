# 🤖 Sistema Inteligente de Predicción de Dígitos Manuscritos

Este proyecto consiste en un sistema completo basado en inteligencia artificial capaz de predecir dígitos manuscritos a partir de imágenes. Utiliza una API desarrollada en **Python con FastAPI** y un frontend visual en **.NET Core MVC**, conectado a un modelo entrenado con **TensorFlow y Keras**.

---

## 🌐 Enlaces del Proyecto

🔹 **API en Render:**  
https://api-prediccion-digitos-2.onrender.com/predict-digit/

🔹 **Aplicación MVC en Azure (.NET):**  
https://predicciontitanic.azurewebsites.net/ *(disponible temporalmente)*

---

## 📷 Capturas de la Aplicación

### Interfaz Web (.NET MVC)
![image](https://github.com/user-attachments/assets/8b96116b-ae5c-4025-a518-2f0786b51d30)
![image](https://github.com/user-attachments/assets/ac8e50a3-10e4-4b4b-9700-d39742fb5629)
![image](https://github.com/user-attachments/assets/8632d35d-3d4e-4423-9cf6-f9abeec9346a)


> Estas capturas muestran el flujo de carga de la imagen, predicción y visualización del dígito detectado con su nivel de confianza.

---

## 🧠 Tecnologías Utilizadas

| Componente        | Tecnología              |
|------------------|-------------------------|
| Backend API       | FastAPI + Uvicorn       |
| IA y modelo       | TensorFlow + Keras      |
| Procesamiento     | OpenCV + PIL (Pillow)   |
| Interfaz web      | .NET Core MVC + Bootstrap 5 |
| Despliegue        | Render (API) + Azure (Frontend) |
| Entorno de desarrollo | Visual Studio 2022 + VS Code |

---

## ⚙️ Estructura del Proyecto

```bash
api-prediccion-digitos-2/
├── API.py                       # Código principal de la API en FastAPI
├── digit_model_color_augment.h5 # Modelo CNN entrenado
├── requirements.txt             # Librerías necesarias para el entorno Python
