#!/bin/bash

set -e

STACK_NAME="predictive-maintenance-stack"
REGION="us-east-1"
KEY_PAIR_NAME="your-key-pair"
INSTANCE_TYPE="t3.medium"

echo "========================================"
echo "Deploying Predictive Maintenance System"
echo "========================================"

if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed"
    exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

echo "Using AWS Account: $(aws sts get-caller-identity --query Account --output text)"
echo "Region: $REGION"
echo

echo "Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file cloudformation.yml \
    --stack-name $STACK_NAME \
    --parameter-overrides \
        InstanceType=$INSTANCE_TYPE \
        KeyPairName=$KEY_PAIR_NAME \
    --capabilities CAPABILITY_IAM \
    --region $REGION

echo "Getting deployment information..."
INSTANCE_ID=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`InstanceId`].OutputValue' \
    --output text)

APPLICATION_URL=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`ApplicationURL`].OutputValue' \
    --output text)

MODEL_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`ModelBucket`].OutputValue' \
    --output text)

echo
echo "========================================"
echo "Deployment Complete!"
echo "========================================"
echo "Instance ID: $INSTANCE_ID"
echo "Application URL: $APPLICATION_URL"
echo "Model Bucket: $MODEL_BUCKET"
echo
echo "The application may take 5-10 minutes to fully start."
echo "Monitor the deployment:"
echo "  aws ec2 describe-instance-status --instance-ids $INSTANCE_ID --region $REGION"
echo
echo "SSH to instance:"
echo "  ssh -i ~/.ssh/$KEY_PAIR_NAME.pem ubuntu@$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].PublicDnsName' --output text)"
echo
echo "View application logs:"
echo "  ssh -i ~/.ssh/$KEY_PAIR_NAME.pem ubuntu@$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].PublicDnsName' --output text) 'docker-compose logs -f'"
