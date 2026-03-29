FROM openenv-base:latest
COPY . /app
WORKDIR /app
RUN pip install -e .
CMD ["uvicorn", "server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]