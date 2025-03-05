# Sử dụng Python 3.9
FROM python:3.9

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy toàn bộ file vào container
COPY . /app

# Cài đặt thư viện
RUN pip install --no-cache-dir fastapi uvicorn torch transformers

# Mở cổng 8000
EXPOSE 8000

# Chạy API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
