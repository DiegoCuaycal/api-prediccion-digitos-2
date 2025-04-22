from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

# Cargar el modelo una sola vez al inicio
# model = load_model("model_Mnist_LeNet.h5")
model = load_model("digit_model_color_augment.h5")


def preprocess_custom_image(file_bytes):
    # Leer imagen como array RGB
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = np.array(img)

    # 1) Convertir a escala de grises (si es necesario)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Umbral Otsu invertido
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) Encontrar el contorno más grande
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No se encontraron dígitos en la imagen.")
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    digit = thresh[y:y + h, x:x + w]

    # 4) Redimensionar a 20x20 manteniendo proporción
    scale = 20.0 / max(w, h)
    digit_resized = cv2.resize(digit, (int(w * scale), int(h * scale)))

    # 5) Centrar en un canvas de 28x28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    dx = (28 - digit_resized.shape[1]) // 2
    dy = (28 - digit_resized.shape[0]) // 2
    canvas[dy:dy + digit_resized.shape[0], dx:dx +
           digit_resized.shape[1]] = digit_resized

    # 6) Normalizar y convertir a RGB
    canvas = canvas.astype("float32") / 255.0
    canvas_rgb = np.repeat(canvas[..., np.newaxis], 3, axis=-1)  # (28, 28, 3)

    return np.expand_dims(canvas_rgb, axis=0)  # (1, 28, 28, 3)


@app.post("/predict-digit/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        print("\n--- Nueva solicitud recibida ---")
        print(f"Archivo recibido: {file.filename}")

        contents = await file.read()
        image = preprocess_custom_image(contents)

        print("Realizando predicción...")
        prediction = model.predict(image)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        print(
            f"Predicción completa. Dígito: {digit}, Confianza: {confidence:.2%}")

        return JSONResponse(
            content={
                "predicted_digit": digit,
                "confidence": f"{confidence*100:.1f}%"
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
