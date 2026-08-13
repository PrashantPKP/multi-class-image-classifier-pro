# 🖼️ Multi-Class Image Classifier 

> An end-to-end deep learning web application that classifies images into **13 scene and object categories** using Transfer Learning (MobileNetV2) with a FastAPI backend and a modern responsive UI.

[![GitHub Repo](https://img.shields.io/badge/GitHub-multi--class--image--classifier-181717?style=for-the-badge&logo=github)](https://github.com/PrashantPKP/multi-class-image-classifier-pro)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.122-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

---

## 📌 Overview

**Multi-Class Image Classifier** is a production-ready deep learning application built with **TensorFlow, FastAPI, HTML/CSS/JavaScript, and Docker**.

Users can upload an image directly from the browser or paste a public image URL, and the model will predict the scene/object category along with a confidence score in real time.

The model is based on **MobileNetV2** (Transfer Learning) — pre-trained on ImageNet and fine-tuned on a custom dataset expanded from the original Intel Image Classification dataset. It achieves **~95% validation accuracy** across 13 classes.

---

## 🚀 Features

- 🖼️ Upload images from browser (drag & drop or browse)
- 🔗 Predict from a public image URL
- ⚡ Real-time image preview before submission
- 🤖 Deep Learning classification with confidence score
- 📊 Visual confidence progress bar in results
- 🌐 FastAPI REST API backend
- 🐳 Docker containerization
- 🔁 Hot-reload dev server support
- 🛡️ Input validation and graceful error handling
- 📱 Fully responsive web interface

---


## 🧠 Model Details

| Property          | Value                                            |
|-------------------|--------------------------------------------------|
| Base Model        | MobileNetV2 (ImageNet weights)                   |
| Fine-tuned Layers | Last 20 layers of MobileNetV2                    |
| Input Size        | 224 × 224 × 3                                    |
| Output            | Softmax over 13 classes                          |
| Loss              | Sparse Categorical Crossentropy                  |
| Optimizer         | Adam (lr=1e-4 → 1e-5 fine-tune)                  |
| Val Accuracy      | ~95%                                             |
| Model File        | `multi_class_classifier_v1.keras`                |
| Preprocessing     | Baked inside model graph (MobileNetV2 normalize) |

### Training Strategy

1. **Phase 1 – Feature Extraction**: MobileNetV2 base frozen, only the top classification head trained (4 epochs).
2. **Phase 2 – Fine-Tuning**: Last 20 layers of MobileNetV2 unfrozen and trained end-to-end at a lower learning rate.
3. **Class Weighting**: Applied to handle class imbalance across 13 categories.
4. **Callbacks**: `ModelCheckpoint` (best val_accuracy), `ReduceLROnPlateau`, `EarlyStopping`.

### Supported Classes (13)

| #  | Class        | Description                  |
|----|--------------|------------------------------|
| 0  | 🚴 Bike      | Bicycles / cyclists          |
| 1  | ⚽ Ball      | Ball / sports ball           |
| 2  | 🏢 Buildings | Urban buildings / structures |
| 3  | 🚌 Bus       | Buses                        |
| 4  | 🚗 Car       | Cars / vehicles              |
| 5  | 🐱 Cats      | Domestic cats                |
| 6  | 🐕 Dogs      | Domestic dogs                |
| 7  | 🌲 Forest    | Dense forest / trees         |
| 8  | ❄️ Glacier   | Ice / glacial landscapes     |
| 9  | ⛰️ Mountain  | Mountain landscapes          |
| 10 | 🌊 Sea       | Ocean / sea scenes           |
| 11 | 🛣️ Street    | Roads / urban streets        |
| 12 | 🚚 Truck     | Trucks / heavy vehicles      |

---

## 🛠️ Tech Stack

| Layer              | Technology                      |
|--------------------|---------------------------------|
| ML Framework       | TensorFlow 2.x / Keras          |
| Base Model         | MobileNetV2 (Transfer Learning) |
| Backend            | FastAPI + Uvicorn               |
| Frontend           | HTML5, CSS3, Vanilla JavaScript |
| Templating         | Jinja2                          |
| Image Processing   | Pillow, NumPy                   |
| Containerization   | Docker                          |
| Deployment         | Render / Docker / Local         |

---

## 🏃 Getting Started

### Prerequisites

- Python 3.10+
- pip
- (Optional) Docker

### 1. Clone the Repository

```bash
git clone https://github.com/PrashantPKP/multi-class-image-classifier-pro.git
cd multi-class-image-classifier-pro
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

### 4. Open in Browser

```
http://localhost:8000
```

---

## 🐳 Docker Setup

### Build Image

```bash
docker build -t multi-class-image-classifier-pro .
```

### Run Container

```bash
docker run -p 8000:8000 multi-class-image-classifier-pro
```

Open in browser:

```
http://localhost:8000
```

---

## 🔌 REST API Reference

### `POST /api/predict`
Classify an image uploaded as multipart form data.

**Request:**
```
Content-Type: multipart/form-data
Body: file = <image file>
```

**Response:**
```json
{
    "prediction": "forest",
    "confidence": 0.9891,
    "image_path": "/uploads/forest.jpg"
}
```

---

### `POST /api/predict-url`
Classify an image from a public URL.

**Request:**
```
POST /api/predict-url?image_url=https://example.com/image.jpg
```

**Response:**
```json
{
    "prediction": "sea",
    "confidence": 0.8341,
    "image_path": "/uploads/<uuid>.jpg"
}
```

---

### `GET /health`
Health check endpoint.

```json
{
    "status": "ok",
    "service": "Multi-Class Image Classifier API",
    "timestamp": "2026-08-14T00:00:00"
}
```

---

### `GET /admin/model-info`
Returns the loaded model path and class list.

---

### `POST /admin/reload-model`
Hot-reloads the model from disk without restarting the server.

---

## 🏋️ Training Your Own Model

To retrain the model on the existing dataset:

```bash
# Fresh training
python traning_data/train.py --output models/multi_class_classifier_v1.keras --epochs 12

# Resume training from existing checkpoint
python traning_data/train.py --resume --output models/multi_class_classifier_v1.keras --epochs 8 --batch-size 16

# Resume with backup of previous model
python traning_data/train.py --resume --backup --output models/multi_class_classifier_v1.keras --epochs 8
```

Training data must be organized as:
```
traning_data/train/
├── bike/
├── buildings/
├── forest/
├── sea/
└── ... (one folder per class)
```

---

## ⚙️ Requirements

```text
fastapi==0.122.0
uvicorn==0.38.0
tensorflow==2.21.0
Jinja2==3.1.6
python-multipart==0.0.20
numpy==2.2.5
pillow==11.2.1
requests==2.32.5
```


## 👨‍💻 Author

**Prashant Parshuramkar**
- GitHub: [@PrashantPKP](https://github.com/PrashantPKP)
- LinkedIn: [prashantpkp](https://www.linkedin.com/in/prashantpkp/)

---

## ⭐ Star This Project

If you found this project useful or interesting, consider giving it a ⭐ on [GitHub](https://github.com/PrashantPKP/multi-class-image-classifier-pro) — it really helps!
