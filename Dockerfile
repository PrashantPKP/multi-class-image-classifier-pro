FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Verify the model file is present at build time — fails fast if it's missing
RUN test -f models/multi_class_classifier_v1.keras \
    && echo "✅ Model file verified: $(du -sh models/multi_class_classifier_v1.keras)" \
    || (echo "❌ ERROR: models/multi_class_classifier_v1.keras not found in image!" && exit 1)

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
