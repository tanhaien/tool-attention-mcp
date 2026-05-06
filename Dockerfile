FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir -e .

ENV TA_USE_TOOL_ATTENTION=0
CMD ["tool-attention-mcp"]
