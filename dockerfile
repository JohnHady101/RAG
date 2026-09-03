FROM python:3.13

WORKDIR /app

# Install system dependencies and set up virtual environment
RUN apt update && \
    apt install -y python3-venv && \
    python3 -m venv .venv

COPY src/requirements.txt ./requirements.txt


# Activate virtual environment and install Python requirements
# Note: We need to ensure the venv's pip is used
RUN . .venv/bin/activate && \
    pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Ensure the virtual environment is activated for subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

