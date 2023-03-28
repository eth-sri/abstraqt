#!/bin/bash

# enable bash strict mode
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail


# navigate to the directory containing this script
cd "$(dirname "$0")"

CONFIG=$(pwd)/config
rm -f "$CONFIG/config.toml"

# according to .bashrc
SOCK="${DOCKER_HOST:7}"

HOSTNAME=$(cat /proc/sys/kernel/hostname)

docker run -d --name gitlab-runner --restart always \
	-v $CONFIG:/etc/gitlab-runner \
	-v $SOCK:/var/run/docker.sock \
	gitlab/gitlab-runner:latest

docker run \
		--rm \
		-it \
		-v $CONFIG:/etc/gitlab-runner \
		--memory 16g \
	gitlab/gitlab-runner register \
		--tag-list "" \
		--name $HOSTNAME-runner \
		--url "https://gitlab.inf.ethz.ch/" \
		--registration-token "************" \
		--executor "docker" \
		--docker-image "ubuntu:20.04" \
		--non-interactive

sed "s/concurrent = .*/concurrent = 20/g" -i $CONFIG/config.toml
