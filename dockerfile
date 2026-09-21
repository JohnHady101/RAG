FROM python:3.13

WORKDIR /app

# Install system dependencies and set up virtual environment
RUN apt update 


COPY ./requirements.txt ./requirements.txt

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


# Activate virtual environment and install Python requirements
# Note: We need to ensure the venv's pip is used
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --break-system-packages -r requirements.txt; fi

# Ensure the virtual environment is activated for subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

