USER_NAME := perception
TAG_NAME := v1
IMAGE_NAME := efficient_instance_pred

WANDB_API_KEY := $(shell echo $$WANDB_API_KEY)
UID := $(shell id -u)
GID := $(shell id -g)
NUSCENES_PATH := /home/robesafe/nuscenes/trainval

define run_docker
	docker run -it --rm \
		--net host \
		--gpus all \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(IMAGE_NAME)_container \
		-u $(USER_NAME) \
		-v ./:/home/$(USER_NAME)/workspace \
		-v $(NUSCENES_PATH):/home/$(USER_NAME)/Datasets/nuscenes \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		'$(IMAGE_NAME)':$(TAG_NAME) \
		/bin/bash -c $(1)
endef

build:
	docker build ./docker -t '$(IMAGE_NAME)':$(TAG_NAME) --force-rm --build-arg USER=$(USER_NAME) --build-arg USER_ID=$(UID) --build-arg USER_GID=$(GID)

attach:
	docker exec -it $(IMAGE_NAME)_container bash

run:
	$(call run_docker, "bash")


