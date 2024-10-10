FROM python:3.10

COPY gvi.py gvi.py
RUN pip install marimo matplotlib  --no-cache-dir
RUN pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
CMD [ "marimo", "run", "app.py", "--host", "0.0.0.0", "-p", "8080" ]