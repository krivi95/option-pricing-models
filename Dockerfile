FROM python:3.7

COPY . /app

WORKDIR /app

RUN pip install -r Requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]

CMD ["streamlit_app.py"]