# Start with a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy the API code and the models folder into the container
COPY src/deployment/ /app/
COPY models/hybrid/ /app/models/
COPY models/hybrid_classifier/ /app/models/

# Expose the port that the API will run on
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]