FROM ubuntu:latest
LABEL authors="ndong"

ENTRYPOINT ["top", "-b"]

# Use a slim Python 3.11 base image
FROM python:3.12.10-slim

# Install uv (package installer)
RUN pip install uv

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (for build cache efficiency)
COPY requirements.txt .

# Install dependencies with uv
RUN uv pip install --system -r requirements.txt

# Copy the rest of your project files (code, .env, etc)
COPY . .

# (Optional: ensure Python output prints immediately)
ENV PYTHONUNBUFFERED=1

# Default command to run your script
EXPOSE 8888
CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]