IMAGE ?= airaneel/babash
TAG ?= latest
PLATFORM ?= linux/amd64
DOCKER_USER ?= airaneel
DOCKER_TOKEN ?= $(DOCKERHUB_TOKEN)

.PHONY: build push login all

build:
	docker buildx build --platform $(PLATFORM) -t $(IMAGE):$(TAG) --load .

login:
	@echo $(DOCKER_TOKEN) | docker login -u $(DOCKER_USER) --password-stdin

push: login
	docker buildx build --platform $(PLATFORM) -t $(IMAGE):$(TAG) --push .

all: push
