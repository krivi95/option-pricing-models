FROM python:3.7

COPY . /app

WORKDIR /app

RUN pip install -r Requirements.txt

EXPOSE 8080

CMD streamlit run --server.port 8080 --server.enableCORS false streamlit_app.py