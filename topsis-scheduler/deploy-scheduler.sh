#!/bin/bash
# deploy-scheduler.sh - Complete deployment script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Kubernetes cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        print_error "Make sure your cluster is running and kubectl is configured"
        exit 1
    fi
    
    # Check if Maven is available
    if ! command -v mvn &> /dev/null; then
        print_error "Maven is not installed or not in PATH"
        exit 1
    fi
    
    print_success "All prerequisites met"
}

# Build the application
build_application() {
    print_status "Building Enhanced Multi-Criteria Scheduler..."
    
    if [ -f "pom.xml" ]; then
        mvn clean package -q
        if [ $? -eq 0 ]; then
            print_success "Application built successfully"
        else
            print_error "Failed to build application"
            exit 1
        fi
    else
        print_error "pom.xml not found. Run this script from the project root directory."
        exit 1
    fi
}

# Build Docker image
build_docker_image() {
    print_status "Building Docker image..."
    
    # Create Dockerfile if it doesn't exist
    if [ ! -f "Dockerfile" ]; then
        print_status "Creating Dockerfile..."
        cat > Dockerfile << 'EOF'
FROM openjdk:17-jre-slim

WORKDIR /app

# Copy the JAR file
COPY target/enhanced-scheduler.jar /app/enhanced-scheduler.jar

# Create logs directory
RUN mkdir -p /app/logs

# Set JVM options for better container performance
ENV JAVA_OPTS="-Xms512m -Xmx1g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ps aux | grep '[j]ava.*enhanced-scheduler' || exit 1

# Run the scheduler
CMD ["java", "-jar", "/app/enhanced-scheduler.jar"]
EOF
    fi
    
    docker build -t enhanced-scheduler:latest .
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    print_status "Deploying to Kubernetes..."
    
    # Apply manifests in order
    print_status "Creating namespace..."
    kubectl apply -f - << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: scheduler-system
  labels:
    name: scheduler-system
EOF

    print_status "Creating RBAC resources..."
    kubectl apply -f k8s-manifests/rbac.yaml
    
    print_status "Creating ConfigMap..."
    kubectl apply -f k8s-manifests/configmap.yaml
    
    print_status "Deploying scheduler..."
    kubectl apply -f k8s-manifests/deployment.yaml
    
    print_status "Creating service..."
    kubectl apply -f k8s-manifests/service.yaml
    
    print_status "Creating test namespaces..."
    kubectl apply -f k8s-manifests/test-namespaces.yaml
    
    print_success "Deployment completed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    print_status "Waiting for scheduler to be ready..."
    
    kubectl wait --for=condition=available --timeout=300s deployment/enhanced-scheduler -n scheduler-system
    
    if [ $? -eq 0 ]; then
        print_success "Scheduler is ready!"
    else
        print_error "Scheduler failed to become ready within 5 minutes"
        print_status "Check logs with: kubectl logs -n scheduler-system deployment/enhanced-scheduler"
        exit 1
    fi
}

# Deploy test workloads
deploy_test_workloads() {
    print_status "Deploying test workloads..."
    
    kubectl apply -f k8s-manifests/test-workloads.yaml
    kubectl apply -f k8s-manifests/priority-classes.yaml
    
    print_success "Test workloads deployed"
}

# Show status
show_status() {
    print_status "Deployment Status:"
    echo ""
    
    print_status "Scheduler Pod Status:"
    kubectl get pods -n scheduler-system -l app=enhanced-scheduler
    echo ""
    
    print_status "Test Pods Status:"
    kubectl get pods -n scheduler-test
    echo ""
    
    print_status "Recent Events:"
    kubectl get events -n scheduler-test --sort-by='.firstTimestamp' | tail -10
}

# Show logs
show_logs() {
    print_status "Recent Scheduler Logs:"
    kubectl logs -n scheduler-system deployment/enhanced-scheduler --tail=50
}

# Cleanup function
cleanup() {
    print_status "Cleaning up deployment..."
    
    kubectl delete namespace scheduler-test --ignore-not-found=true
    kubectl delete namespace scheduler-monitoring --ignore-not-found=true
    kubectl delete namespace scheduler-system --ignore-not-found=true
    kubectl delete priorityclass high-priority-workload low-priority-workload --ignore-not-found=true
    kubectl delete clusterrole enhanced-scheduler --ignore-not-found=true
    kubectl delete clusterrolebinding enhanced-scheduler --ignore-not-found=true
    
    print_success "Cleanup completed"
}

# Main function
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            build_application
            build_docker_image
            deploy_kubernetes
            wait_for_deployment
            deploy_test_workloads
            show_status
            print_success "Enhanced Multi-Criteria Scheduler deployed successfully!"
            print_status "Monitor with: ./deploy-scheduler.sh monitor"
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "monitor")
            while true; do
                clear
                show_status
                echo ""
                show_logs
                sleep 30
            done
            ;;
        "test")
            print_status "Running algorithm tests..."
            # Scale up load generator to test Weighted Sum
            kubectl scale deployment load-generator --replicas=60 -n scheduler-test
            print_status "Scaled load generator to 60 replicas to trigger Weighted Sum algorithm"
            sleep 10
            show_logs | grep -E "(TOPSIS|VIKOR|Weighted Sum|High load|conflict)"
            ;;
        "cleanup")
            cleanup
            ;;
        "help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy   - Build and deploy the scheduler (default)"
            echo "  status   - Show deployment status"
            echo "  logs     - Show recent scheduler logs"
            echo "  monitor  - Continuous monitoring (Ctrl+C to exit)"
            echo "  test     - Run algorithm selection tests"
            echo "  cleanup  - Remove all deployed resources"
            echo "  help     - Show this help message"
            ;;
        *)
            print_error "Unknown command: $1"
            print_status "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"