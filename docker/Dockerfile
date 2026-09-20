FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY contextlab ./contextlab
RUN pip install --no-cache-dir .
COPY . .
ENTRYPOINT ["contextlab"]
CMD ["doctor"]
