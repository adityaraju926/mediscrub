FROM python:3.10-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords')"

# Copy all project files to /app directory
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app (ui.py will be at /app/ui.py)
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.enableCORS=false"] 