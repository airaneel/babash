IMAGE ?= airaneel/babash
TAG ?= latest
DOCKER_USER ?= airaneel
DOCKER_TOKEN ?= $(DOCKERHUB_TOKEN)

.PHONY: build push login all

build:
	docker build -t $(IMAGE):$(TAG) .

login:
	@echo $(DOCKER_TOKEN) | docker login -u $(DOCKER_USER) --password-stdin

push: login build
	docker push $(IMAGE):$(TAG)

all: push
