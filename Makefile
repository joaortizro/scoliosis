# ── Config ────────────────────────────────────────────────────────────────────
EC2_HOST      ?= ec2-user@<your-ec2-ip>
EC2_APP       ?= /home/ec2-user/scoliosis
PEM           ?= ~/.ssh/your-key.pem
DIST_DIR      := dist

# ECR / ECS config
AWS_REGION    ?= us-east-1
AWS_ACCOUNT   ?= <your-aws-account-id>
ECR_REPO      ?= scoliosis-api
ECR_URI       := $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO)
ECS_CLUSTER   ?= scoliosis-cluster
ECS_SERVICE   ?= scoliosis-service
IMAGE_TAG     ?= $(shell cat VERSION | tr -d '[:space:]')

# ── Local pipeline ────────────────────────────────────────────────────────────
.PHONY: preprocess train evaluate pipeline

preprocess:
	dvc repro preprocess

train:
	dvc repro train

evaluate:
	dvc repro evaluate

pipeline:
	dvc repro

# ── Package ───────────────────────────────────────────────────────────────────
.PHONY: build clean

clean:
	rm -rf $(DIST_DIR) *.egg-info

build: clean
	python -m build --wheel --outdir $(DIST_DIR)
	@echo "Built: $$(ls $(DIST_DIR)/*.whl)"

# ── Push model checkpoint to S3 (via DVC) ─────────────────────────────────────
.PHONY: push-data

push-data:
	dvc push

# ── Deploy to EC2 ─────────────────────────────────────────────────────────────
.PHONY: deploy deploy-whl deploy-server

# Copy wheel + server code to EC2
deploy-whl: build
	scp -i $(PEM) $(DIST_DIR)/*.whl $(EC2_HOST):$(EC2_APP)/dist/

# Restart the FastAPI server on EC2 after installing the new wheel
deploy-server: deploy-whl
	ssh -i $(PEM) $(EC2_HOST) " \
		cd $(EC2_APP) && \
		pip install --force-reinstall dist/*.whl && \
		dvc pull && \
		sudo systemctl restart scoliosis-server \
	"

# Full deploy: build wheel + push data + deploy server
deploy: push-data deploy-server

# ── ECR / ECS deploy (production path) ───────────────────────────────────────
.PHONY: ecr-login docker-build docker-push ecs-deploy deploy-ecs

# Authenticate Docker to ECR
ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_URI)

# Build Docker image tagged with VERSION
docker-build:
	docker build -t $(ECR_REPO):$(IMAGE_TAG) .
	docker tag $(ECR_REPO):$(IMAGE_TAG) $(ECR_URI):$(IMAGE_TAG)
	docker tag $(ECR_REPO):$(IMAGE_TAG) $(ECR_URI):latest

# Push image to ECR
docker-push: ecr-login docker-build
	docker push $(ECR_URI):$(IMAGE_TAG)
	docker push $(ECR_URI):latest

# Trigger rolling deployment on ECS (zero-downtime)
ecs-deploy:
	aws ecs update-service \
		--cluster $(ECS_CLUSTER) \
		--service $(ECS_SERVICE) \
		--force-new-deployment \
		--region $(AWS_REGION)
	@echo "Deployment triggered. Monitor at:"
	@echo "https://$(AWS_REGION).console.aws.amazon.com/ecs/home#/clusters/$(ECS_CLUSTER)/services"

# Full ECS deploy: push data + build image + push to ECR + redeploy service
deploy-ecs: push-data docker-push ecs-deploy

# ── Helpers ───────────────────────────────────────────────────────────────────
.PHONY: ssh mlflow-ui test

ssh:
	ssh -i $(PEM) $(EC2_HOST)

mlflow-ui:
	bash mlflow/mlflow_server.sh

test:
	tox -e test_package
