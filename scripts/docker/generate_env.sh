
# Call with:
# docker run --rm --workdir /$USER -v /home/vlugli/homeGPU/datcom-tfm:/$USER -e HOME=/$USER ngpu_docker bash ./scripts/docker/generate_env.sh

python3.8 -m venv docker_env
source docker_env/bin/activate
pip install -r ./scripts/docker/requirements_ngpu_docker.txt
