FROM python:3.9

RUN mkdir /app
RUN mkdir /datasets
WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data data
COPY model model
COPY scripts scripts
COPY tools tools
COPY config.ini .
COPY main.py .
COPY multiple_plot_centres_file_config.py .
COPY entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]