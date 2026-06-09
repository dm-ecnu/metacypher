# MetaCypher — Application Dockerfile
#
# Build:   docker build -t metacypher .
# Run:     docker run --rm metacypher
#
# The default CMD runs the offline smoke test (no Neo4j or vLLM required).
# For real pipeline runs, override CMD and set NEO4J_HOST / NEO4J_USER /
# NEO4J_PASSWORD / METACYPHER_VLLM_BASE_URL via -e or docker-compose.

FROM python:3.10-slim

# Keep Python output unbuffered so log lines appear immediately.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy dependency manifest first so Docker can cache the install layer.
COPY requirements.txt ./

# Install only the non-heavy dependencies; torch/faiss/sentence-transformers
# are optional (used by FAISS attribute matching) and excluded here to keep
# the image small.  Add them back if you need the embedding-based search path.
RUN pip install --no-cache-dir \
        neo4j==5.28.1 \
        networkx==3.4.2 \
        "numpy>=1.26,<2" \
        tqdm==4.67.1 \
        "openai>=1.59,<2"

# Copy the full repository.
COPY . .

# Default: run the offline smoke test from inside the package directory,
# exactly as documented in examples/smoke_offline.py.
CMD ["sh", "-c", "cd metacypher && python ../examples/smoke_offline.py"]
