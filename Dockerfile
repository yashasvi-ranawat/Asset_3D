FROM python:3.10
LABEL prod "prediction"
EXPOSE 8000
ENV PROJECT_DIR /usr/local/src/webapp
COPY app ${PROJECT_DIR}  
COPY Pipfile ${PROJECT_DIR}/Pipfile  
WORKDIR ${PROJECT_DIR}
RUN ["pip", "install", "pipenv"]
RUN ["python", "-m", "pipenv", "install", "--deploy"]
RUN ["python", "-m", "pipenv", "run", "python", "download_models.py"]
ENTRYPOINT  ["python", "-m", "pipenv", "run", "uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]  
