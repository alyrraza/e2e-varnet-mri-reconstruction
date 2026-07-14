FROM python:3.11-slim

WORKDIR /app

# System deps for h5py and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libhdf5-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install fastMRI from pinned source commit first (not pip)
RUN git clone https://github.com/facebookresearch/fastMRI.git /tmp/fastmri \
    && cd /tmp/fastmri \
    && git checkout 91f2df4711adbb6d643df1810f234e4abcf5881b \
    && pip install -e . --no-cache-dir \
    && rm -rf /tmp/fastmri/.git

# Install project dependencies
COPY mlops/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code and assets
COPY mlops/ /app/mlops/
COPY checkpoints/ /app/checkpoints/

# Make src importable as a package from /app
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "mlops.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
