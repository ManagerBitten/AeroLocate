FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 aerolocate
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ARG INSTALL_ML=true
RUN if [ "$INSTALL_ML" = "true" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir -r requirements-ml.txt; \
    fi
COPY backend ./backend
COPY scripts ./scripts
RUN mkdir -p /data/flight_datasets && chown -R aerolocate:aerolocate /data
USER aerolocate
ENV AEROLOCATE_DATA_DIR=/data
ENV AEROLOCATE_LOCAL_FLIGHT_DATASET_DIR=/flight_datasets
ENV HF_HOME=/data/huggingface
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
