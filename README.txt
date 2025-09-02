# Start from Python image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Install Node.js (LTS) + npm
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PostgreSQL client (optional but useful)
RUN apt-get install -y postgresql-client

# Default working directory
WORKDIR /workspace



POSTGRES_DB=citizenoffice
POSTGRES_USER=citizen
POSTGRES_PASSWORD=citizen
DJANGO_SECRET_KEY=changeme
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_USER=changeme
EMAIL_PASS=changeme



















