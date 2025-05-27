#!/usr/bin/env bash
./stop_vllm_server.sh

python vllm_server.py "$@"
