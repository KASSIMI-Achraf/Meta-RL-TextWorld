@echo off
REM Run a command inside the TextWorld Docker container
REM Usage: docker_run.bat <command>
REM Example: docker_run.bat python experiments/run_experiment.py --mode train

docker exec -it my-textworld /bin/bash -c "cd /workspace && %*"
