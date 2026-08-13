import time
import requests

URL = "https://intel-image-classification-fastapi.onrender.com/health"

while True:

    try:

        response = requests.get(URL)

        print("Health Check:", response.status_code)

    except Exception as e:

        print("Error:", e)

    # Wait 12 minutes
    time.sleep(12 * 60)