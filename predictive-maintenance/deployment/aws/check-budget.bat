@echo off
echo ============================================
echo PyTorch Predictive Maintenance - Budget Monitor
echo Account: 974136094538
echo Region: eu-west-2
echo ============================================

echo 1. Current Budget Status...
aws budgets describe-budget --account-id 974136094538 --budget-name predictive-maintenance-pytorch-budget --query "Budget.{Name:BudgetName,Limit:BudgetLimit.Amount,Unit:BudgetLimit.Unit}" --output table 2>nul || echo No budget found

echo 2. Current Month Spending...
aws ce get-dimension-values --time-period Start=2025-08-01,End=2025-08-31 --dimension SERVICE --query "DimensionValues[?contains(Value, 'Container')].{Service:Value}" --output table

echo 3. ECS Service Status...
aws ecs describe-services --cluster predictive-maintenance-pytorch-cluster --services predictive-maintenance-pytorch-service --region eu-west-2 --query "services[0].{ServiceName:serviceName,Status:status,DesiredCount:desiredCount,RunningCount:runningCount}" --output table 2>nul || echo No services found

echo 4. Running Tasks Count...
for /f "tokens=*" %%i in ('aws ecs list-tasks --cluster predictive-maintenance-pytorch-cluster --region eu-west-2 --query "taskArns" --output text') do (
    if NOT "%%i"=="None" (
        echo Active tasks found - incurring compute charges
    ) else (
        echo No active tasks - no compute charges
    )
)

echo 5. ECR Repository Size...
aws ecr describe-repositories --repository-names predictive-maintenance-pytorch --region eu-west-2 --query "repositories[0].{Name:repositoryName,Size:repositorySizeInBytes}" --output table 2>nul || echo Repository not found

echo 6. Cost Estimate Calculation...
echo.
echo COST BREAKDOWN:
echo ================
echo ECR Storage: ~$0.15/month (2GB PyTorch image)
echo ECS Fargate: 
for /f "tokens=*" %%i in ('aws ecs list-tasks --cluster predictive-maintenance-pytorch-cluster --region eu-west-2 --query "taskArns" --output text') do (
    if NOT "%%i"=="None" (
        echo   - RUNNING: ~$7/month after free tier
        echo   - FREE TIER: $0/month first year
    ) else (
        echo   - STOPPED: $0/month
    )
)
echo Load Balancer: ~$18/month (if created)
echo Total Current: Check CloudWatch billing dashboard

echo 7. Recent Billing Alerts...
aws cloudwatch describe-alarms --alarm-names "pytorch-predictive-maintenance-cost-alarm" --region us-east-1 --query "MetricAlarms[0].{Name:AlarmName,State:StateValue,Threshold:Threshold}" --output table 2>nul || echo No billing alarms found

echo 8. Quick Actions Available...
echo.
echo EMERGENCY COMMANDS:
echo - Stop immediately: emergency-stop-pytorch.bat
echo - Check this status: check-budget.bat  
echo - View costs: aws ce get-cost-and-usage (manual)
echo.
echo WEB DASHBOARDS:
echo - AWS Console: https://eu-west-2.console.aws.amazon.com/ecs/
echo - Billing: https://console.aws.amazon.com/billing/
echo - CloudWatch: https://eu-west-2.console.aws.amazon.com/cloudwatch/

echo ============================================
echo Budget Monitor Complete
echo Run this script daily to track spending
echo ============================================
pause