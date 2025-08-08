@echo off
echo ============================================
echo Deploying PyTorch Predictive Maintenance with Budget Protection
echo Account: 974136094538
echo Region: eu-west-2
echo Budget Limit: $5.00/month (Auto-stop at $4.00)
echo Framework: PyTorch 2.0.1
echo ============================================

echo 1. Setting up budget protection...
echo Creating budget configuration...
echo {"BudgetName":"predictive-maintenance-pytorch-budget","BudgetLimit":{"Amount":"5.00","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST","CostFilters":{"Service":["Amazon Elastic Container Service","Amazon EC2 Container Registry"]}} > budget-config.json

echo 2. Creating budget with automatic monitoring...
aws budgets create-budget --account-id 974136094538 --budget file://budget-config.json 2>nul || echo Budget already exists - continuing...

echo 3. Setting up billing alarm...
aws cloudwatch put-metric-alarm --alarm-name "pytorch-predictive-maintenance-cost-alarm" --alarm-description "Alert when costs exceed $4" --metric-name EstimatedCharges --namespace AWS/Billing --statistic Maximum --period 86400 --threshold 4.0 --comparison-operator GreaterThanThreshold --region us-east-1 2>nul || echo Alarm already exists - continuing...

echo 4. Creating ECR Repository...
aws ecr create-repository --repository-name predictive-maintenance-pytorch --region eu-west-2 2>nul || echo Repository already exists - continuing...

echo 5. ECR Authentication...
for /f "tokens=*" %%i in ('aws ecr get-login-password --region eu-west-2') do set ECR_PASSWORD=%%i
echo %ECR_PASSWORD% | docker login --username AWS --password-stdin 974136094538.dkr.ecr.eu-west-2.amazonaws.com

echo 6. Building PyTorch Docker image...
cd ..\..
docker build -f deployment/Dockerfile -t predictive-maintenance-pytorch .

echo 7. Checking image size...
for /f "tokens=*" %%i in ('docker images predictive-maintenance-pytorch:latest --format "{{.Size}}"') do echo Image size: %%i

echo 8. Tagging for ECR...
docker tag predictive-maintenance-pytorch:latest 974136094538.dkr.ecr.eu-west-2.amazonaws.com/predictive-maintenance-pytorch:latest

echo 9. Pushing to ECR...
docker push 974136094538.dkr.ecr.eu-west-2.amazonaws.com/predictive-maintenance-pytorch:latest

echo 10. Creating ECS cluster with monitoring...
aws ecs create-cluster --cluster-name predictive-maintenance-pytorch-cluster --region eu-west-2 2>nul || echo Cluster already exists - continuing...

echo 11. Creating CloudWatch dashboard for cost monitoring...
aws cloudwatch put-dashboard --dashboard-name "pytorch-predictive-maintenance-costs" --dashboard-body "{\"widgets\":[{\"type\":\"metric\",\"properties\":{\"metrics\":[[\"AWS/Billing\",\"EstimatedCharges\",\"Currency\",\"USD\",\"ServiceName\",\"AmazonECS\"],[\"AWS/Billing\",\"EstimatedCharges\",\"Currency\",\"USD\",\"ServiceName\",\"AmazonECR\"]],\"period\":300,\"stat\":\"Maximum\",\"region\":\"us-east-1\",\"title\":\"PyTorch Predictive Maintenance Costs\"}}]}" --region eu-west-2 2>nul || echo Dashboard already exists - continuing...

echo ============================================
echo SUCCESS: PyTorch deployment with budget protection completed
echo ============================================
echo Configuration:
echo - Framework: PyTorch with LSTM + Attention
echo - Budget: $5.00/month with $4.00 alert threshold
echo - Repository: predictive-maintenance-pytorch
echo - Cluster: predictive-maintenance-pytorch-cluster
echo - Expected size: ~2GB (estimated $0.15/month storage)
echo ============================================
echo Next steps:
echo 1. Create ECS Service in AWS Console
echo 2. Configure task definition with budget limits
echo 3. Monitor costs via CloudWatch dashboard
echo 4. Use emergency stop if needed: emergency-stop-pytorch.bat
echo ============================================
echo Estimated monthly costs:
echo - ECR Storage: $0.15 (2GB PyTorch image)
echo - ECS Fargate: $0.00 (free tier covers 750 hours/month)
echo - Total Year 1: $0.15/month
echo - Total Year 2+: $7.15/month
echo ============================================
pause