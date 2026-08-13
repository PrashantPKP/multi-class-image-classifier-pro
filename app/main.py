from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.predict import predict_image, reload_model, get_model_path, CLASS_NAMES
from datetime import datetime
import shutil
import requests
import uuid
import os

os.makedirs("uploads", exist_ok=True)

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)

templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...)
):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = predict_image(file_path)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": f"{confidence:.4f}",
            "image_path": f"/uploads/{file.filename}"
        }
    )


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = predict_image(file_path)

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "image_path": f"/uploads/{file.filename}"
    }


@app.post("/api/predict-url")
async def predict_url(image_url: str):

    try:

        # Create unique filename

        filename = (
            f"uploads/{uuid.uuid4()}.jpg"
        )

        # Download image

        response = requests.get(
            image_url,
            timeout=10,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        # Verify content is image

        content_type = (
            response.headers.get(
                "Content-Type",
                ""
            )
        )

        if not content_type.startswith(
            "image/"
        ):

            return {
                "error":
                "Invalid URL. Please provide a direct image URL (.jpg, .jpeg, .png, .webp)"
            }

        # Save image

        with open(
            filename,
            "wb"
        ) as f:

            f.write(
                response.content
            )

        # Predict

        prediction, confidence = (
            predict_image(filename)
        )

        return {

            "prediction":
            prediction,

            "confidence":
            float(confidence),

            "image_path":
            "/" + filename

        }

    except requests.exceptions.Timeout:

        return {
            "error":
            "Request timed out while downloading image."
        }

    except requests.exceptions.ConnectionError:

        return {
            "error":
            "Unable to connect to image URL."
        }

    except requests.exceptions.HTTPError:

        return {
            "error":
            "Image URL returned an invalid response."
        }

    except Exception as e:

        print(
            "Prediction Error:",
            str(e)
        )

        return {
            "error":
            str(e)
        }

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": "Multi-Class Image Classifier API",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/admin/reload-model")
async def admin_reload_model():
    try:
        selected = reload_model()
        return {"status": "reloaded", "model_path": str(selected)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/admin/model-info")
async def admin_model_info():
    path = get_model_path()
    return {
        "model_path": str(path) if path is not None else None,
        "classes": CLASS_NAMES,
    }