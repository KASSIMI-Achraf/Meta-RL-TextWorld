#!/bin/bash
# Run a command inside the TextWorld Docker container
# Usage: ./docker_run.sh <command>
# Example: ./docker_run.sh python experiments/run_experiment.py --mode train

docker exec -it my-textworld /bin/bash -c "cd /workspace && $*"
