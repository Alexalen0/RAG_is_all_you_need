# Use python:3.8-slim as the base image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt to the Docker image
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files to the Docker image
COPY . .

# Run the Streamlit application
CMD ["streamlit", "run", "openvino.py"]
