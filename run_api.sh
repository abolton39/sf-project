#!/bin/bash

docker build -t model_api .
docker run -d -p 1313:1313 --name model_api_container model_api
