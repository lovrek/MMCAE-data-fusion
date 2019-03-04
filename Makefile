.PHONY: help

docker-build-image:
	docker build -t mag .

start:
	NV_GPU=2 docker run -d --name=mag_container --runtime=nvidia -v /home/lpodgorsek/mag:/mag -v /home/lpodgorsek/scratch/data:/data -p 127.0.0.1:8888:8888 -p 127.0.0.1:6006:6006 --env TENSORBOARD_LOGDIR="/mag/logs/" mag:latest

stop:
	docker stop mag_container && docker rm mag_container

bash:
	docker exec -it mag_container /bin/bash

log:
	docker logs mag_container --follow --tail 100

screen:
	screen -D -R -S mag
