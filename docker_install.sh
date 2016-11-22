#!/bin/bash

echo "This script will attempt to install docker service on your linux machine. Run script with root privileges."

echo "deb https://apt.dockerproject.org/repo debian-jessie main" > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install docker-engine
service docker start

