FROM python:3.11-slim

# Install git and other build deps just in case
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory to the standard Hugging Face directory
WORKDIR /code

# Copy all the current files from the repo root to /code
COPY . /code

# Install the project locally along with OpenEnv
RUN pip install --no-cache-dir -e .

# Hugging Face explicitly requires the app to listen on port 7860
EXPOSE 7860

# Run Uvicorn with factory mode
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
