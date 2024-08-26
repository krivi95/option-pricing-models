FROM python:3.9

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r Requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "--server.port", "8080", "--server.address", "0.0.0.0", "streamlit_app.py"]

# build command
# docker build -t options-pricing:latest .

# run command
# docker run -p 8080:8080 options-pricing:latest