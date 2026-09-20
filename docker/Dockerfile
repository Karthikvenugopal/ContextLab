FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir .
ENTRYPOINT ["contextlab"]
CMD ["doctor"]

FROM base AS development
RUN pip install --no-cache-dir ".[dev,analysis]"
