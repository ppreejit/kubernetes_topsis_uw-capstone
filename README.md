# kubernetes_topsis_uw-capstone
# TOPSIS Scheduler for Kubernetes

## Overview
TOPSIS Scheduler is an energy-aware and performance-optimized custom Kubernetes scheduler that implements the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) multi-criteria decision-making method. This capstone project demonstrates how multi-criteria decision-making can enhance Kubernetes scheduling to optimize for both performance and energy efficiency.

The scheduler extends Kubernetes' resource management capabilities by considering multiple criteria when placing pods, including:

- Task execution time
- Energy consumption
- Available CPU resources
- Available memory
- Resource balance across nodes

### Project Structure

This capstone project consists of three main components:

1. **TOPSIS Scheduler**: A custom Kubernetes scheduler implementing the TOPSIS algorithm (Java)
2. **Linear Regression Workloads**: Three test applications with varying resource requirements:
   - Small linear regression (basic workload)
   - Scalable linear regression (medium complexity)
   - Distributed linear regression (high complexity)
3. **Metrics Collector**: A utility to gather and analyze scheduling decisions and performance metrics

## Key Features

- **Multi-criteria Decision Making**: Uses TOPSIS algorithm to find optimal node placement based on weighted criteria
- **Energy Awareness**: Estimates and minimizes energy consumption for workload execution
- **Detailed Metrics**: Tracks and stores scheduling performance metrics for analysis
- **Resource Optimization**: Considers both resource availability and balance when making scheduling decisions
- **Smart Task Analysis**: Analyzes tasks based on complexity (lines of code) and workload type

## How It Works

1. The scheduler identifies unscheduled pods that have been assigned to the `topsis-scheduler` scheduler
2. It evaluates all available nodes in the cluster based on multiple criteria
3. The TOPSIS algorithm calculates an optimal node for each pod based on weighted preferences
4. Metrics about the scheduling decision are recorded as pod annotations
5. The pod is bound to the selected node

## Prerequisites

- Java 17 or higher
- Docker Desktop
- Google Kubernetes Engine cluster 
- Google Cloud SDK Shell
- Kubernetes API access with sufficient permissions to:
  - List and watch pods
  - List and watch nodes
  - Create pod bindings
  - Create events
  - Patch pod annotations

## Installation

### 1. Build and Deploy the TOPSIS Scheduler

#### Build the Project
```bash
# Navigate to the topsis-scheduler directory
cd topsis-scheduler

# Build with Maven
mvn clean package

# Build Docker image
docker build -t ppreejit/topsis-scheduler:v1 .

# Push to Docker Hub (or your preferred registry)
docker push ppreejit/topsis-scheduler:v1
```
#### Create a New GCP Project
```bash
# Create a new project
gcloud projects create PROJECT_ID --name="PROJECT_NAME"

# Set the new project as your current project
gcloud config set project PROJECT_ID
```

#### Enable Billing for the Project
```bash
# List available billing accounts
gcloud billing accounts list

# Link your billing account to the project
gcloud billing projects link PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
```

#### Enable Required APIs
```bash
# Enable the Kubernetes Engine API and related services
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
```

#### Create a GKE Cluster for Testing
```bash
# Create the base cluster with a minimal pool for system components
gcloud container clusters create topsis-test-cluster \
  --num-nodes=1 \
  --machine-type=e2-standard-2 \
  --disk-size=100 \
  --cluster-version=latest \
  --no-enable-autoupgrade \
  --enable-ip-alias \
  --tags=topsis-test

# Create Category A node pool with Spot VMs (low-resource nodes)
gcloud container node-pools create category-a-pool \
  --cluster=topsis-test-cluster \
  --machine-type=e2-medium \
  --num-nodes=2 \
  --disk-size=50 \
  --node-labels=node-category=category-a,cloud.google.com/gke-spot=true \
  --spot \
  --node-taints=cloud.google.com/gke-spot=true:NoSchedule

# Create Category B node pool with Spot VMs (medium-resource nodes)
gcloud container node-pools create category-b-pool \
  --cluster=topsis-test-cluster \
  --machine-type=e2-standard-2 \
  --num-nodes=1 \
  --disk-size=80 \
  --node-labels=node-category=category-b,cloud.google.com/gke-spot=true \
  --spot \
  --node-taints=cloud.google.com/gke-spot=true:NoSchedule

# Create Category C node pool with Spot VMs (high-resource nodes)
gcloud container node-pools create category-c-pool \
  --cluster=topsis-test-cluster \
  --machine-type=n2-standard-4 \
  --num-nodes=1 \
  --disk-size=100 \
  --node-labels=node-category=category-c,cloud.google.com/gke-spot=true \
  --spot \
  --node-taints=cloud.google.com/gke-spot=true:NoSchedule

# Verify node pools are created
kubectl get nodes --show-labels
```

### 2. Deploy the TOPSIS Scheduler to Kubernetes

#### Create RBAC Resources
Save the following to `rbac.yaml`:

```yaml
# serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: topsis-scheduler
  namespace: kube-system
---
# clusterrole.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: topsis-scheduler
rules:
- apiGroups: [""]
  resources: ["pods", "nodes", "events"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: [""]
  resources: ["pods/binding"]
  verbs: ["create"]
- apiGroups: [""]
  resources: ["pods/status"]
  verbs: ["patch", "update"]
---
# clusterrolebinding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: topsis-scheduler
subjects:
- kind: ServiceAccount
  name: topsis-scheduler
  namespace: kube-system
roleRef:
  kind: ClusterRole
  name: topsis-scheduler
  apiGroup: rbac.authorization.k8s.io
```

Apply the RBAC configuration:
```bash
kubectl apply -f rbac.yaml
```

#### Create Deployment
Save the following to `topsis-scheduler-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: topsis-scheduler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: topsis-scheduler
  template:
    metadata:
      labels:
        app: topsis-scheduler
    spec:
      serviceAccountName: topsis-scheduler
      tolerations:
      - key: "cloud.google.com/gke-spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
      - name: topsis-scheduler
        image: ppreejit/topsis-scheduler:v1
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        env:
        - name: INSTRUCTIONS_PER_LOC
          value: "10"
```

Apply the deployment:
```bash
kubectl apply -f topsis-scheduler-deployment.yaml
```

Verify the scheduler is running:
```bash
kubectl get deployments -n kube-system
kubectl get pods -n kube-system
```

### 3. Deploy Metrics Collector

#### Create RBAC resources for the metrics collector
First, create the necessary RBAC resources by saving the following to `metrics-agent-rbac.yaml`:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: metrics-agent-sa
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: metrics-agent-role
rules:
- apiGroups: [""]
  resources: ["pods", "nodes"]
  verbs: ["get", "watch", "list"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: metrics-agent-role-binding
subjects:
- kind: ServiceAccount
  name: metrics-agent-sa
  namespace: default
roleRef:
  kind: ClusterRole
  name: metrics-agent-role
  apiGroup: rbac.authorization.k8s.io
```

Apply the RBAC configuration:
```bash
kubectl apply -f metrics-agent-rbac.yaml
```

#### Deploy the metrics collector pod
Save the following to `metrics-collector-pod.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: metrics-collector
  namespace: default
spec:
  serviceAccountName: metrics-agent-sa
  containers:
  - name: collector
    image: ubuntu:20.04
    command:
    - /bin/bash
    - -c
    # Long bash script included in the file
    resources:
      limits:
        cpu: "200m"
        memory: "512Mi"
      requests:
        cpu: "100m"
        memory: "256Mi"
    volumeMounts:
    - name: metrics-storage
      mountPath: /tmp
  volumes:
  - name: metrics-storage
    emptyDir: {}
```

Deploy the metrics collector:
```bash
kubectl apply -f metrics-collector-pod.yaml
```

Verify the metrics collector is running:
```bash
kubectl get pods
```

## Usage

### 1. Building and Deploying Test Workloads

The project includes three different types of linear regression workloads that can be used to test the scheduler:

- Small (basic workload)
- Scalable (medium complexity workload)
- Large/Distributed (high complexity workload)

#### Build Test Workloads
```bash
# Navigate to the linear-regression directory
cd linear-regression

# Build with Maven
mvn clean package

# Build Docker images for each workload type
docker build -f Dockerfile.small -t ppreejit/linear-regression-small:latest .
docker build -f Dockerfile.scalable -t ppreejit/linear-regression-scalable:latest .
docker build -f Dockerfile.large -t ppreejit/linear-regression-large:latest .

# Push images to Docker Hub
docker push ppreejit/linear-regression-small:latest
docker push ppreejit/linear-regression-scalable:latest
docker push ppreejit/linear-regression-large:latest
```

### 2. Scheduling Pods with TOPSIS Scheduler

To schedule a pod using the TOPSIS scheduler, specify `schedulerName: topsis-scheduler` in the pod specification. You'll also need to add tolerations for the Spot VMs if using the GKE setup described above.

Here's an example deployment for each type of workload:

#### Small Workload
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: regression-small
  annotations:
    task.type: "io_intensive"  
    task.size: "small"
    task.loc: "400"
spec:
  schedulerName: topsis-scheduler
  tolerations:
  - key: "cloud.google.com/gke-spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  containers:
  - name: regression
    image: ppreejit/linear-regression-small:latest
    resources:
      requests:
        cpu: 200m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi
```

#### Scalable Workload
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: regression-scalable
  annotations:
    task.type: "compute_intensive"
    task.size: "scalable"
    task.loc: "850"
spec:
  schedulerName: topsis-scheduler
  tolerations:
  - key: "cloud.google.com/gke-spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  containers:
  - name: regression
    image: ppreejit/linear-regression-scalable:latest
    resources:
      requests:
        cpu: 500m
        memory: 512Mi
      limits:
        cpu: 1000m
        memory: 1Gi
```

#### Large/Distributed Workload
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: regression-large
  annotations:
    task.type: "compute_intensive"
    task.size: "distributed"
    task.loc: "1200"
spec:
  schedulerName: topsis-scheduler
  tolerations:
  - key: "cloud.google.com/gke-spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  containers:
  - name: regression
    image: ppreejit/linear-regression-large:latest
    resources:
      requests:
        cpu: 800m
        memory: 1Gi
      limits:
        cpu: 1500m
        memory: 2Gi
```

Deploy the workloads:
```bash
kubectl apply -f regression-small.yaml
kubectl apply -f regression-scalable.yaml
kubectl apply -f regression-large.yaml
```
You can see which scheduler was used to schedule a pod using the following command:
kubectl describe pod <pod_name>

## Sample events for a pod scheduled using topsis-scheduler
Events:
  Type    Reason     Age        From              Message
  ----    ------     ----       ----              -------
  Normal  Scheduled  <unknown>  topsis-scheduler  Pod scheduled by topsis-scheduler
  Normal  Pulling    6m2s       kubelet           Pulling image "ppreejit/small-regression:latest"
  Normal  Pulled     5m44s      kubelet           Successfully pulled image "ppreejit/small-regression:latest" in 18.284s (18.284s including waiting). Image size: 408072718 bytes.
  Normal  Created    5m44s      kubelet           Created container: small-regression
  Normal  Started    5m43s      kubelet           Started container small-regression

## Sample events for a pod scheduled using default-scheduler
Events:
  Type    Reason     Age    From               Message
  ----    ------     ----   ----               -------
  Normal  Scheduled  8m35s  default-scheduler  Successfully assigned default/small-regression-default-1 to gke-topsis-test-clust-category-b-pool-ca60c57c-jr97
  Normal  Pulling    8m35s  kubelet            Pulling image "ppreejit/small-regression:latest"
  Normal  Pulled     8m18s  kubelet            Successfully pulled image "ppreejit/small-regression:latest" in 17.309s (17.309s including waiting). Image size: 408072718 bytes.
  Normal  Created    8m18s  kubelet            Created container: small-regression
  Normal  Started    8m18s  kubelet            Started container small-regression 

### Task Annotations

You can use the following annotations to provide hints to the scheduler about your workload:

- `task.type`: Specifies the type of workload
  - `compute_intensive`: CPU-bound tasks (applies higher instruction count)
  - `io_intensive`: I/O-bound tasks (applies lower instruction count)
  - `memory_intensive`: Memory-bound tasks (applies medium instruction count)
  
- `task.size`: Indicates the scaling nature of the workload
  - `small`: Standard workload (base scaling factor)
  - `scalable`: Workload that scales with resources (50% more instructions)
  - `distributed`: Highly parallel workload (150% more instructions)
  
- `task.loc`: Estimated lines of code in the workload (used to estimate complexity)

- `task.source`: Actual source code (if available, will count non-empty, non-comment lines)

- `task.github`: GitHub repository information (for future expansion)

### Viewing Scheduling Metrics
To view the scheduling metrics collected by the metrics-collector pod:

```bash
# View all metrics
kubectl logs metrics-collector

# If you want to follow the logs in real-time (similar to tail -f), you can add the -f flag:
kubectl logs -f metrics-collector
```
The metrics CSV file includes:

1. Timestamp of the measurement
2. Pod name and namespace
3. Scheduler used (TOPSIS or default)
4. Node selection
5. Scheduling time in milliseconds
6. Energy consumption estimates (J and kJ)

## Algorithm Details

### TOPSIS Method

The TOPSIS method follows these steps:

1. **Create decision matrix**: Evaluate each node against all criteria
2. **Normalize matrix**: Scale values to make them comparable
3. **Apply weights**: Apply importance weights to each criterion
4. **Determine ideal solutions**: Find the best and worst possible values for each criterion
5. **Calculate separations**: Compute distances from ideal and negative-ideal solutions
6. **Calculate relative closeness**: Determine how close each alternative is to the ideal solution
7. **Rank alternatives**: Select the node with the highest relative closeness

### Criteria Weights

Default weights are as follows (can be modified in the code):

- Execution Time: 0.2
- Energy Consumption: 0.2
- Available CPU: 0.2
- Available Memory: 0.2
- Resource Balance: 0.2

## Performance Considerations

- The scheduler polls for unscheduled pods every 10 seconds
- API timeouts are set to 60 seconds
- Error handling includes exponential backoff for API failures
- Performance metrics are logged and stored as pod annotations
- Detailed timing breakdowns are available in the logs

## Experiment Design and Testing

### Testing Scenarios

The project includes three predefined test scenarios with varying levels of resource competition to evaluate the TOPSIS scheduler against the default Kubernetes scheduler:

1. **Low Competition (Baseline Test)**
   - 2 small regression pods with TOPSIS scheduler
   - 2 small regression pods with default scheduler
   - 1 scalable regression pod with TOPSIS scheduler
   - 1 scalable regression pod with default scheduler
   - 1 distributed regression pod with TOPSIS scheduler
   - 1 distributed regression pod with default scheduler

2. **Medium Competition (Moderate Resource Contention)**
   - 4 small regression pods with TOPSIS scheduler
   - 4 small regression pods with default scheduler
   - 2 scalable regression pods with TOPSIS scheduler
   - 2 scalable regression pods with default scheduler
   - 1 distributed regression pod with TOPSIS scheduler
   - 1 distributed regression pod with default scheduler

3. **High Competition (Heavy Resource Contention)**
   - 6 small regression pods with TOPSIS scheduler
   - 6 small regression pods with default scheduler
   - 3 scalable regression pods with TOPSIS scheduler
   - 3 scalable regression pods with default scheduler
   - 2 distributed regression pods with TOPSIS scheduler
   - 2 distributed regression pods with default scheduler

### Running the Tests

To run the tests, apply the corresponding YAML file:

```bash
# For low competition test
kubectl apply -f low-competition.yaml

# For medium competition test
kubectl apply -f medium-competition.yaml

# For high competition test
kubectl apply -f high-competition.yaml
```

To clean up between tests:
```bash
kubectl delete pods --all -n default
```

### Workload Specifications

Each workload type has specific resource requirements and characteristics:

1. **Small Regression**
   - CPU Request: 100m
   - Memory Request: 256Mi
   - Lines of Code: 110
   - Task Type: compute_intensive
   - Task Size: small

2. **Scalable Regression**
   - CPU Request: 200m
   - Memory Request: 384Mi
   - Lines of Code: 185
   - Task Type: compute_intensive
   - Task Size: scalable
   - Parameters: BATCH_SIZE=10000, NUM_THREADS=4

3. **Distributed Regression**
   - CPU Request: 500m
   - Memory Request: 1Gi
   - Lines of Code: 435
   - Task Type: compute_intensive
   - Task Size: distributed
   - Parameters: Multiple parameters for partitioning and threading

You can use this data to compare:
1. Scheduling efficiency (time to schedule pods)
2. Energy consumption estimates
3. Node selection patterns
4. Resource utilization balance

## Troubleshooting

### Common Issues

1. **Pods remain in "Pending" state**:
   - Verify that pods specify `schedulerName: topsis-scheduler`
   - Check scheduler logs for errors: `kubectl logs -n kube-system -l app=topsis-scheduler`
   - Ensure all nodes have the "Ready" condition
   - Check that tolerations match the node taints (e.g., `cloud.google.com/gke-spot`)

2. **"No eligible nodes found" error**:
   - Pods may request more resources than available
   - Check node resource usage with `kubectl describe nodes`
   - Verify that nodes have appropriate labels for your scheduling criteria

3. **Scheduling latency issues**:
   - For large clusters, consider increasing connection timeouts
   - Review timing metrics in the scheduler logs
   
4. **Metrics collector not working**:
   - Check the service account permissions
   - Verify the pod is running with `kubectl get pod metrics-collector`
   - Check logs with `kubectl logs metrics-collector`

## Results and Findings

The results have been documented and submitted along with all the deliverables.

## Conclusion

This capstone project demonstrates the implementation and evaluation of a multi-criteria decision-making approach to Kubernetes pod scheduling. By leveraging the TOPSIS method, we've created a scheduler that considers multiple factors including energy consumption, which is increasingly important in modern cloud environments.

The experimental results show that a more sophisticated scheduling approach can provide benefits in resource utilization and energy efficiency compared to the default Kubernetes scheduler, especially in heterogeneous cluster environments with varying workload types.

Future work could explore dynamic weight adjustment based on cluster conditions, integration with Kubernetes metrics server for real-time monitoring, and expansion to additional scheduling criteria such as network topology or data locality.