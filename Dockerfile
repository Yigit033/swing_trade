FROM python:3.11-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Bağımlılıkları kopyala ve yükle (layer cache için önce)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kaynak kodu kopyala
COPY . .

# Streamlit config (headless mod)
RUN mkdir -p /root/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = 8501\n\
address = 0.0.0.0\n\
" > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "swing_trader/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
