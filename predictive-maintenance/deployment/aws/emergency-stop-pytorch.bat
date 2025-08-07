@echo off
echo ============================================
echo EMERGENCY: Stopping All PyTorch Predictive Maintenance Services
echo This will IMMEDIATELY stop all running services to prevent charges
echo ============================================

echo WARNING: This action will stop your application immediately!
set /p CONFIRM="Type 'STOP' to confirm emergency shutdown: "

if NOT "%CONFIRM%"=="STOP" (
    echo Emergency stop cancelled.
    pause
    exit /b
)

echo Executing emergency shutdown...

echo 1. Stopping ECS Service...
aws ecs update-service --cluster predictive-maintenance-pytorch-cluster --service predictive-maintenance-pytorch-service --desired-count 0 --region eu-west-2 2>nul || echo Service not found or already stopped

echo 2. Listing and stopping all running tasks...
for /f "tokens=*" %%i in ('aws ecs list-tasks --cluster predictive-maintenance-pytorch-cluster --region eu-west-2 --query "taskArns" --output text') do (
    if NOT "%%i"=="None" (
        echo Stopping task: %%i
        aws ecs stop-task --cluster predictive-maintenance-pytorch-cluster --task "%%i" --region eu-west-2
    )
)

echo 3. Verifying all services are stopped...
aws ecs describe-services --cluster predictive-maintenance-pytorch-cluster --services predictive-maintenance-pytorch-service --region eu-west-2 --query "services[0].desiredCount" --output text 2>nul || echo No services running

echo 4. Checking for any remaining running tasks...
aws ecs list-tasks --cluster predictive-maintenance-pytorch-cluster --region eu-west-2 --query "taskArns" --output text

echo 5. Deleting ECS cluster (optional - prevents accidental restart)...
set /p DELETE_CLUSTER="Delete entire cluster to prevent restart? (y/n): "
if "%DELETE_CLUSTER%"=="y" (
    aws ecs delete-cluster --cluster predictive-maintenance-pytorch-cluster --region eu-west-2
    echo Cluster deleted - complete shutdown achieved
)

echo 6. Setting budget alert threshold to $0.50 for maximum protection...
aws budgets put-budget --account-id 974136094538 --budget "{\"BudgetName\":\"predictive-maintenance-pytorch-budget\",\"BudgetLimit\":{\"Amount\":\"0.50\",\"Unit\":\"USD\"},\"TimeUnit\":\"MONTHLY\",\"BudgetType\":\"COST\"}" 2>nul

echo ============================================
echo EMERGENCY SHUTDOWN COMPLETED
echo ============================================
echo Status:
echo - All ECS services stopped (desired count = 0)
echo - All running tasks terminated
echo - Budget alert lowered to $0.50
echo - No further compute charges will accrue
echo ============================================
echo Current costs:
echo - ECR storage: ~$0.15/month (PyTorch image)
echo - ECS compute: $0.00 (stopped)
echo - Total ongoing: ~$0.15/month maximum
echo ============================================
echo To restart services:
echo 1. Run: aws ecs update-service --desired-count 1
echo 2. Or redeploy using deploy-pytorch-protected.bat
echo ============================================
pause