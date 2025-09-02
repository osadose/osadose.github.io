{
  "name": "CitizenOffice",
  "dockerFile": "Dockerfile",
  "forwardPorts": [8000, 3000],
  "postCreateCommand": "pip install -r backend/requirements.txt || true && npm install --prefix frontend || true",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.isort",
        "esbenp.prettier-vscode",
        "dbaeumer.vscode-eslint"
      ]
    }
  }
}




# Start from Python image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Install Node.js + npm
RUN apt-get update && apt-get install -y nodejs npm

# Upgrade pip
RUN pip install --upgrade pip

# Default working directory
WORKDIR /workspace




# Node
node_modules/
.next/
npm-debug.log
yarn-error.log




version: '3.9'
services:
  db:
    image: postgis/postgis:15-3.4
    environment:
      POSTGRES_DB: citizenoffice
      POSTGRES_USER: citizen
      POSTGRES_PASSWORD: citizen
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"



