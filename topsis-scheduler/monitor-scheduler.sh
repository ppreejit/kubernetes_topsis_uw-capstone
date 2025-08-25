#!/bin/bash
# monitor-scheduler.sh - Real-time monitoring and analysis

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Print functions
print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}        ${BLUE}Enhanced Multi-Criteria Scheduler Monitor${NC}        ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo -e "\n${YELLOW}▶ $1${NC}"
    echo -e "${YELLOW}$(printf '%.0s─' $(seq 1 ${#1}))${NC}"
}

print_metric() {
    printf "%-25s: %s\n" "$1" "$2"
}

# Get scheduler pod name
get_scheduler_pod() {
    kubectl get pods -n scheduler-system -l app=enhanced-scheduler -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo ""
}

# Check if scheduler is running
check_scheduler_health() {
    local pod_name=$(get_scheduler_pod)
    if [ -z "$pod_name" ]; then
        echo -e "${RED}❌ Scheduler pod not found${NC}"
        return 1
    fi
    
    local status=$(kubectl get pod $pod_name -n scheduler-system -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$status" == "Running" ]; then
        echo -e "${GREEN}✅ Scheduler is running${NC}"
        return 0
    else
        echo -e "${RED}❌ Scheduler status: $status${NC}"
        return 1
    fi
}

# Get algorithm statistics from logs
get_algorithm_stats() {
    local pod_name=$(get_scheduler_pod)
    if [ -z "$pod_name" ]; then
        return 1
    fi
    
    print_section "Algorithm Usage Statistics"
    
    # Get recent logs
    local logs=$(kubectl logs $pod_name -n scheduler-system --tail=1000 2>/dev/null)
    
    # Count algorithm usage
    local topsis_count=$(echo "$logs" | grep -c "Selected.*TOPSIS" || echo "0")
    local vikor_count=$(echo "$logs" | grep -c "Selected.*VIKOR" || echo "0")
    local weighted_count=$(echo "$logs" | grep -c "Selected.*WEIGHTED_SUM" || echo "0")
    local total=$((topsis_count + vikor_count + weighted_count))
    
    if [ $total -gt 0 ]; then
        local topsis_pct=$(echo "scale=1; $topsis_count * 100 / $total" | bc -l 2>/dev/null || echo "0")
        local vikor_pct=$(echo "scale=1; $vikor_count * 100 / $total" | bc -l 2>/dev/null || echo "0")
        local weighted_pct=$(echo "scale=1; $weighted_count * 100 / $total" | bc -l 2>/dev/null || echo "0")
        
        print_metric "TOPSIS" "${GREEN}$topsis_count${NC} (${topsis_pct}%)"
        print_metric "VIKOR" "${BLUE}$vikor_count${NC} (${vikor_pct}%)"
        print_metric "Weighted Sum" "${YELLOW}$weighted_count${NC} (${weighted_pct}%)"
        print_metric "Total Decisions" "$total"
    else
        print_metric "Status" "${YELLOW}No scheduling decisions recorded yet${NC}"
    fi
    
    # Show recent algorithm selections
    echo ""
    echo -e "${CYAN}Recent Algorithm Selections:${NC}"
    echo "$logs" | grep -E "(Selected.*algorithm|High load detected|criteria conflict)" | tail -5 | while read line; do
        if echo "$line" | grep -q "TOPSIS"; then
            echo -e "  ${GREEN}→${NC} $line"
        elif echo "$line" | grep -q "VIKOR"; then
            echo -e "  ${BLUE}→${NC} $line"
        elif echo "$line" | grep -q "WEIGHTED_SUM"; then
            echo -e "  ${YELLOW}→${NC} $line"
        else
            echo -e "  ${PURPLE}→${NC} $line"
        fi
    done
}

# Get scheduling performance metrics
get_performance_metrics() {
    local pod_name=$(get_scheduler_pod)
    if [ -z "$pod_name" ]; then
        return 1
    fi
    
    print_section "Performance Metrics"
    
    local logs=$(kubectl logs $pod_name -n scheduler-system --tail=500 2>/dev/null)
    
    # Extract timing metrics
    local avg_total=$(echo "$logs" | grep "Total scheduling time:" | tail -10 | sed 's/.*: \([0-9]*\) ms.*/\1/' | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "0"}')
    local avg_algorithm=$(echo "$logs" | grep "calculation completed.*in.*ms" | tail -10 | sed 's/.*in \([0-9]*\) ms.*/\1/' | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "0"}')
    local avg_binding=$(echo "$logs" | grep "Binding time:" | tail -10 | sed 's/.*: \([0-9]*\) ms.*/\1/' | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "0"}')
    
    print_metric "Avg Total Time" "${avg_total} ms"
    print_metric "Avg Algorithm Time" "${avg_algorithm} ms"
    print_metric "Avg Binding Time" "${avg_binding} ms"
    
    # Extract energy metrics
    local avg_energy=$(echo "$logs" | grep "Energy consumption:" | tail -10 | sed 's/.*: \([0-9.]*\) kJ.*/\1/' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "0"}')
    
    print_metric "Avg Energy Consumption" "${avg_energy} kJ"
}

# Get pod scheduling status
get_pod_status() {
    print_section "Pod Scheduling Status"
    
    # Test namespace pods
    local test_pods=$(kubectl get pods -n scheduler-test --no-headers 2>/dev/null | wc -l)
    local running_pods=$(kubectl get pods -n scheduler-test --no-headers 2>/dev/null | grep -c "Running" || echo "0")
    local pending_pods=$(kubectl get pods -n scheduler-test --no-headers 2>/dev/null | grep -c "Pending" || echo "0")
    local failed_pods=$(kubectl get pods -n scheduler-test --no-headers 2>/dev/null | grep -c "Failed" || echo "0")
    
    print_metric "Total Test Pods" "$test_pods"
    print_metric "Running" "${GREEN}$running_pods${NC}"
    print_metric "Pending" "${YELLOW}$pending_pods${NC}"
    print_metric "Failed" "${RED}$failed_pods${NC}"
    
    # Show pods with scheduler annotations
    echo ""
    echo -e "${CYAN}Pods with Enhanced Scheduler Annotations:${NC}"
    kubectl get pods -n scheduler-test -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName,ALGORITHM:.metadata.annotations.scheduler\\.algorithm,SCORE:.metadata.annotations.scheduler\\.score --no-headers 2>/dev/null | while read line; do
        if echo "$line" | grep -q "TOPSIS"; then
            echo -e "  ${GREEN}→${NC} $line"
        elif echo "$line" | grep -q "VIKOR"; then
            echo -e "  ${BLUE}→${NC} $line"
        elif echo "$line" | grep -q "WEIGHTED_SUM"; then
            echo -e "  ${YELLOW}→${NC} $line"
        else
            echo -e "  → $line"
        fi
    done
}

# Get cluster resource utilization
get_cluster_resources() {
    print_section "Cluster Resource Utilization"
    
    # Get node information
    local nodes=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
    local ready_nodes=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready " || echo "0")
    
    print_metric "Total Nodes" "$nodes"
    print_metric "Ready Nodes" "${GREEN}$ready_nodes${NC}"
    
    # Get resource requests vs capacity
    echo ""
    echo -e "${CYAN}Node Resource Summary:${NC}"
    kubectl top nodes 2>/dev/null | tail -n +2 | while read node cpu_pct cpu_abs mem_pct mem_abs; do
        echo -e "  ${BLUE}$node${NC}: CPU ${cpu_pct} Memory ${mem_pct}"
    done 2>/dev/null || echo "  Resource metrics not available (metrics-server required)"
}

# Get recent events
get_recent_events() {
    print_section "Recent Scheduling Events"
    
    # Show recent events from test namespace
    kubectl get events -n scheduler-test --sort-by='.firstTimestamp' 2>/dev/null | tail -5 | while read line; do
        if echo "$line" | grep -q "Scheduled"; then
            echo -e "  ${GREEN}→${NC} $line"
        elif echo "$line" | grep -q "Failed"; then
            echo -e "  ${RED}→${NC} $line"
        else
            echo -e "  → $line"
        fi
    done
}

# Get detailed scheduler logs
get_detailed_logs() {
    local pod_name=$(get_scheduler_pod)
    if [ -z "$pod_name" ]; then
        return 1
    fi
    
    print_section "Recent Scheduler Logs"
    
    kubectl logs $pod_name -n scheduler-system --tail=20 2>/dev/null | while read line; do
        if echo "$line" | grep -q "ERROR\|SEVERE"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -q "WARN"; then
            echo -e "${YELLOW}$line${NC}"
        elif echo "$line" | grep -q "TOPSIS\|VIKOR\|WEIGHTED_SUM"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -q "Selected.*algorithm"; then
            echo -e "${BLUE}$line${NC}"
        else
            echo "$line"
        fi
    done
}

# Generate scheduling analysis report
generate_report() {
    local pod_name=$(get_scheduler_pod)
    if [ -z "$pod_name" ]; then
        echo "Scheduler pod not found. Cannot generate report."
        return 1
    fi
    
    local report_file="scheduler-analysis-$(date +%Y%m%d-%H%M%S).txt"
    
    echo "Enhanced Multi-Criteria Scheduler Analysis Report" > $report_file
    echo "Generated: $(date)" >> $report_file
    echo "=================================================" >> $report_file
    echo "" >> $report_file
    
    # Get comprehensive logs for analysis
    kubectl logs $pod_name -n scheduler-system --tail=2000 >> $report_file 2>/dev/null
    
    echo "Report saved to: $report_file"
    
    # Generate summary statistics
    local logs=$(kubectl logs $pod_name -n scheduler-system --tail=2000 2>/dev/null)
    local total_schedules=$(echo "$logs" | grep -c "Successfully bound pod" || echo "0")
    local avg_time=$(echo "$logs" | grep "Total scheduling time:" | sed 's/.*: \([0-9]*\) ms.*/\1/' | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "0"}')
    
    echo ""
    echo -e "${GREEN}Analysis Summary:${NC}"
    print_metric "Total Pods Scheduled" "$total_schedules"
    print_metric "Average Scheduling Time" "${avg_time} ms"
    print_metric "Report File" "$report_file"
}

# Test algorithm selection
test_algorithms() {
    print_section "Testing Algorithm Selection"
    
    echo "1. Testing normal load (should use TOPSIS/VIKOR)..."
    kubectl apply -f - << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: test-normal-load
  namespace: scheduler-test
  annotations:
    task.type: "compute_intensive"
    task.priority: "normal"
    task.loc: "500"
spec:
  schedulerName: "enhanced-scheduler"
  containers:
  - name: test
    image: nginx:alpine
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
  restartPolicy: Never
EOF

    sleep 5
    
    echo "2. Testing high load scenario (should use Weighted Sum)..."
    kubectl scale deployment load-generator --replicas=60 -n scheduler-test
    
    sleep 10
    
    echo "3. Checking recent algorithm selections..."
    local pod_name=$(get_scheduler_pod)
    if [ ! -z "$pod_name" ]; then
        kubectl logs $pod_name -n scheduler-system --tail=50 | grep -E "(Selected.*algorithm|High load|conflict)" | tail -10
    fi
    
    echo ""
    echo "4. Scaling back load generator..."
    kubectl scale deployment load-generator --replicas=0 -n scheduler-test
    
    echo -e "${GREEN}Algorithm test completed!${NC}"
}

# Cleanup test resources
cleanup_tests() {
    print_section "Cleaning up test resources"
    
    kubectl delete pod test-normal-load -n scheduler-test --ignore-not-found=true
    kubectl scale deployment load-generator --replicas=0 -n scheduler-test
    
    echo -e "${GREEN}Test cleanup completed${NC}"
}

# Main monitoring dashboard
monitor_dashboard() {
    while true; do
        clear
        print_header
        
        # Check if scheduler is healthy
        if ! check_scheduler_health; then
            echo ""
            echo -e "${RED}Scheduler is not healthy. Check deployment status.${NC}"
            echo "Press Ctrl+C to exit or wait for auto-retry..."
            sleep 10
            continue
        fi
        
        get_algorithm_stats
        get_performance_metrics
        get_pod_status
        get_cluster_resources
        get_recent_events
        
        echo ""
        echo -e "${CYAN}Refreshing in 30 seconds... (Press Ctrl+C to exit)${NC}"
        sleep 30
    done
}

# Interactive menu
show_menu() {
    clear
    print_header
    echo ""
    echo "Select monitoring option:"
    echo ""
    echo "1) Real-time Dashboard"
    echo "2) Algorithm Statistics"
    echo "3) Performance Metrics"
    echo "4) Pod Status"
    echo "5) Recent Logs"
    echo "6) Generate Analysis Report"
    echo "7) Test Algorithm Selection"
    echo "8) Cleanup Test Resources"
    echo "9) Exit"
    echo ""
    read -p "Enter choice [1-9]: " choice
    
    case $choice in
        1) monitor_dashboard ;;
        2) clear; print_header; get_algorithm_stats; read -p "Press Enter to continue..." ;;
        3) clear; print_header; get_performance_metrics; read -p "Press Enter to continue..." ;;
        4) clear; print_header; get_pod_status; read -p "Press Enter to continue..." ;;
        5) clear; print_header; get_detailed_logs; read -p "Press Enter to continue..." ;;
        6) clear; print_header; generate_report; read -p "Press Enter to continue..." ;;
        7) clear; print_header; test_algorithms; read -p "Press Enter to continue..." ;;
        8) clear; print_header; cleanup_tests; read -p "Press Enter to continue..." ;;
        9) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid choice. Press Enter to try again..."; read ;;
    esac
}

# Main function
main() {
    case "${1:-menu}" in
        "dashboard"|"monitor")
            monitor_dashboard
            ;;
        "stats"|"statistics")
            clear; print_header; get_algorithm_stats
            ;;
        "performance"|"perf")
            clear; print_header; get_performance_metrics
            ;;
        "status")
            clear; print_header; check_scheduler_health; get_pod_status
            ;;
        "logs")
            clear; print_header; get_detailed_logs
            ;;
        "report")
            clear; print_header; generate_report
            ;;
        "test")
            clear; print_header; test_algorithms
            ;;
        "cleanup")
            clear; print_header; cleanup_tests
            ;;
        "menu"|"")
            while true; do
                show_menu
            done
            ;;
        "help")
            echo "Enhanced Multi-Criteria Scheduler Monitor"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  dashboard  - Real-time monitoring dashboard"
            echo "  stats      - Show algorithm usage statistics"
            echo "  performance- Show performance metrics"
            echo "  status     - Show scheduler and pod status"
            echo "  logs       - Show recent scheduler logs"
            echo "  report     - Generate detailed analysis report"
            echo "  test       - Test algorithm selection"
            echo "  cleanup    - Clean up test resources"
            echo "  menu       - Interactive menu (default)"
            echo "  help       - Show this help"
            ;;
        *)
            echo "Unknown command: $1"
            echo "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Check if bc is available for calculations
if ! command -v bc &> /dev/null; then
    echo "Warning: 'bc' calculator not found. Some statistics may not be accurate."
    echo "Install with: apt-get install bc (Ubuntu/Debian) or brew install bc (macOS)"
fi

# Run main function
main "$@"