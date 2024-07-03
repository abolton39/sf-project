#!/bin/bash

# docker build -t model_api .
# docker run -d -p 1313:1313 --name model_api_container model_api

# Build the Docker image
docker build -t model_api_with_ui .

# Run the Docker container
docker run -d -p 1313:1313 -p 8501:8501 --name model_api_container_with_ui model_api_with_ui
