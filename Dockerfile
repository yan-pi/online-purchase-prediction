FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy>=1.24.3 \
    pandas>=2.0.3 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.7.2 \
    seaborn>=0.12.2 \
    ucimlrepo>=0.0.7 \
    imbalanced-learn>=0.11.0 \
    jupyter>=1.0.0 \
    notebook>=7.0.2

# Copy project files
COPY . .

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
