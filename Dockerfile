FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY ragkit ./ragkit
COPY corpus ./corpus
ENV RAG_CORPUS=/app/corpus
RUN pip install --no-cache-dir .
RUN useradd --create-home appuser
USER appuser
CMD ["python", "-m", "ragkit", "eval"]
