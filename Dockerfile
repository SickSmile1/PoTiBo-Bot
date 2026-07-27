FROM python:3.13-slim

WORKDIR /docbot

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN mkdir -p /data /tmp/codebase-clone

USER root
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-u", "main.py"]
