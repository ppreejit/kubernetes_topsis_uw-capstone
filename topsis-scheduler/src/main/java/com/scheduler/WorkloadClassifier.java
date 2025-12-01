package com.scheduler;

import io.kubernetes.client.openapi.models.*;
import io.kubernetes.client.custom.Quantity;
import java.util.*;
import java.util.logging.Logger;

/**
 * Robust workload classification system that automatically assigns workloads
 * to categories (Light, Scalable, Distributed) based on resource requests,
 * container metadata, and task annotations.
 */
public class WorkloadClassifier {
    private static final Logger logger = Logger.getLogger(WorkloadClassifier.class.getName());

    // Classification thresholds
    private static final double LIGHT_CPU_THRESHOLD = 0.5;          // 500m CPU
    private static final long LIGHT_MEMORY_THRESHOLD = 512 * 1024 * 1024L;  // 512 MB
    private static final double SCALABLE_CPU_THRESHOLD = 2.0;       // 2 CPUs
    private static final long SCALABLE_MEMORY_THRESHOLD = 2L * 1024 * 1024 * 1024;  // 2 GB
    private static final int LIGHT_LOC_THRESHOLD = 500;             // Lines of code
    private static final int SCALABLE_LOC_THRESHOLD = 2000;         // Lines of code

    /**
     * Workload categories
     */
    public enum WorkloadCategory {
        LIGHT,       // Small, single-instance workloads
        SCALABLE,    // Medium workloads that can scale horizontally
        DISTRIBUTED  // Large, complex distributed workloads
    }

    /**
     * Workload type with detailed characteristics
     */
    public enum WorkloadType {
        COMPUTE_INTENSIVE,
        MEMORY_INTENSIVE,
        IO_INTENSIVE,
        ENERGY_EFFICIENT,
        BATCH_PROCESSING,
        REAL_TIME,
        DEFAULT
    }

    /**
     * Classification result containing category, type, and confidence
     */
    public static class ClassificationResult {
        private final WorkloadCategory category;
        private final WorkloadType type;
        private final double confidence;
        private final Map<String, Object> features;
        private final String reasoning;

        public ClassificationResult(WorkloadCategory category, WorkloadType type, 
                                   double confidence, Map<String, Object> features, 
                                   String reasoning) {
            this.category = category;
            this.type = type;
            this.confidence = confidence;
            this.features = features;
            this.reasoning = reasoning;
        }

        public WorkloadCategory getCategory() { return category; }
        public WorkloadType getType() { return type; }
        public double getConfidence() { return confidence; }
        public Map<String, Object> getFeatures() { return features; }
        public String getReasoning() { return reasoning; }
    }

    /**
     * Main classification method that analyzes pod and determines workload category
     */
    public static ClassificationResult classifyWorkload(V1Pod pod) {
        Map<String, Object> features = extractFeatures(pod);
        
        // Check for explicit classification in annotations
        Map<String, String> annotations = pod.getMetadata().getAnnotations();
        if (annotations != null && annotations.containsKey("workload.category")) {
            String explicitCategory = annotations.get("workload.category").toUpperCase();
            try {
                WorkloadCategory category = WorkloadCategory.valueOf(explicitCategory);
                WorkloadType type = determineWorkloadType(annotations, features);
                return new ClassificationResult(category, type, 1.0, features, 
                    "Explicit annotation: workload.category=" + explicitCategory);
            } catch (IllegalArgumentException e) {
                logger.warning("Invalid workload.category annotation: " + explicitCategory);
            }
        }

        // Automatic classification based on features
        return classifyFromFeatures(pod, features);
    }

    /**
     * Extract features from pod specification for classification
     */
    private static Map<String, Object> extractFeatures(V1Pod pod) {
        Map<String, Object> features = new HashMap<>();

        // Resource requests
        double totalCpu = 0;
        long totalMemory = 0;
        long totalStorage = 0;
        int containerCount = 0;

        if (pod.getSpec() != null && pod.getSpec().getContainers() != null) {
            for (V1Container container : pod.getSpec().getContainers()) {
                containerCount++;
                if (container.getResources() != null && 
                    container.getResources().getRequests() != null) {
                    
                    Map<String, Quantity> requests = container.getResources().getRequests();
                    
                    if (requests.containsKey("cpu")) {
                        totalCpu += parseQuantity(requests.get("cpu"));
                    }
                    if (requests.containsKey("memory")) {
                        totalMemory += (long) parseQuantity(requests.get("memory"));
                    }
                    if (requests.containsKey("storage")) {
                        totalStorage += (long) parseQuantity(requests.get("storage"));
                    }
                }
            }
        }

        features.put("cpu_cores", totalCpu);
        features.put("memory_bytes", totalMemory);
        features.put("storage_bytes", totalStorage);
        features.put("container_count", containerCount);

        // CPU to Memory ratio (indicator of workload type)
        double cpuMemoryRatio = totalMemory > 0 ? 
            totalCpu / (totalMemory / (1024.0 * 1024.0 * 1024.0)) : 0;
        features.put("cpu_memory_ratio", cpuMemoryRatio);

        // Annotations analysis
        Map<String, String> annotations = pod.getMetadata().getAnnotations();
        if (annotations != null) {
            // Lines of code
            int linesOfCode = annotations.containsKey("task.loc") ? 
                Integer.parseInt(annotations.getOrDefault("task.loc", "100")) : 100;
            features.put("lines_of_code", linesOfCode);

            // Task type
            String taskType = annotations.getOrDefault("task.type", "default");
            features.put("task_type", taskType);

            // Priority
            String priority = annotations.getOrDefault("task.priority", "normal");
            features.put("priority", priority);

            // Explicit size hint
            String sizeHint = annotations.getOrDefault("task.size", "unknown");
            features.put("size_hint", sizeHint);
        } else {
            features.put("lines_of_code", 100);
            features.put("task_type", "default");
            features.put("priority", "normal");
            features.put("size_hint", "unknown");
        }

        // Labels analysis
        Map<String, String> labels = pod.getMetadata().getLabels();
        if (labels != null) {
            // Check for distributed system indicators
            boolean isDistributed = labels.containsKey("app.kubernetes.io/component") ||
                                   labels.containsKey("statefulset.kubernetes.io/pod-name") ||
                                   labels.containsKey("app") && 
                                   (labels.get("app").contains("kafka") || 
                                    labels.get("app").contains("spark") ||
                                    labels.get("app").contains("hadoop") ||
                                    labels.get("app").contains("cassandra"));
            features.put("is_distributed", isDistributed);

            // Check for batch processing indicators
            boolean isBatch = labels.containsKey("job-name") || 
                             labels.containsKey("batch.kubernetes.io/job-name");
            features.put("is_batch", isBatch);
        } else {
            features.put("is_distributed", false);
            features.put("is_batch", false);
        }

        // Volume mounts (indicator of stateful/persistent workloads)
        int volumeCount = 0;
        boolean hasStatefulVolume = false;
        if (pod.getSpec() != null && pod.getSpec().getVolumes() != null) {
            volumeCount = pod.getSpec().getVolumes().size();
            for (V1Volume volume : pod.getSpec().getVolumes()) {
                if (volume.getPersistentVolumeClaim() != null) {
                    hasStatefulVolume = true;
                    break;
                }
            }
        }
        features.put("volume_count", volumeCount);
        features.put("has_stateful_volume", hasStatefulVolume);

        // Network requirements
        boolean hasHostNetwork = pod.getSpec() != null && 
                                Boolean.TRUE.equals(pod.getSpec().getHostNetwork());
        features.put("has_host_network", hasHostNetwork);

        // Service account (indicates need for K8s API access)
        boolean hasServiceAccount = pod.getSpec() != null && 
                                   pod.getSpec().getServiceAccountName() != null &&
                                   !pod.getSpec().getServiceAccountName().equals("default");
        features.put("has_service_account", hasServiceAccount);

        return features;
    }

    /**
     * Classify workload based on extracted features using decision tree logic
     */
    private static ClassificationResult classifyFromFeatures(V1Pod pod, Map<String, Object> features) {
        StringBuilder reasoning = new StringBuilder();
        List<String> factors = new ArrayList<>();
        
        double cpu = (Double) features.get("cpu_cores");
        long memory = (Long) features.get("memory_bytes");
        int containerCount = (Integer) features.get("container_count");
        int linesOfCode = (Integer) features.get("lines_of_code");
        boolean isDistributed = (Boolean) features.get("is_distributed");
        boolean hasStatefulVolume = (Boolean) features.get("has_stateful_volume");
        String sizeHint = (String) features.get("size_hint");

        WorkloadCategory category;
        double confidence = 0.8; // Base confidence

        // Priority 1: Explicit size hint
        if (!"unknown".equals(sizeHint)) {
            switch (sizeHint.toLowerCase()) {
                case "light":
                case "small":
                    category = WorkloadCategory.LIGHT;
                    confidence = 0.9;
                    factors.add("Explicit size hint: " + sizeHint);
                    break;
                case "scalable":
                case "medium":
                    category = WorkloadCategory.SCALABLE;
                    confidence = 0.9;
                    factors.add("Explicit size hint: " + sizeHint);
                    break;
                case "distributed":
                case "large":
                    category = WorkloadCategory.DISTRIBUTED;
                    confidence = 0.9;
                    factors.add("Explicit size hint: " + sizeHint);
                    break;
                default:
                    category = classifyByResources(cpu, memory, linesOfCode, isDistributed, 
                                                   containerCount, hasStatefulVolume, factors);
            }
        }
        // Priority 2: Distributed system indicators
        else if (isDistributed || containerCount > 3) {
            category = WorkloadCategory.DISTRIBUTED;
            confidence = 0.85;
            if (isDistributed) factors.add("Distributed system labels detected");
            if (containerCount > 3) factors.add("Multiple containers: " + containerCount);
        }
        // Priority 3: Resource-based classification
        else {
            category = classifyByResources(cpu, memory, linesOfCode, isDistributed, 
                                          containerCount, hasStatefulVolume, factors);
        }

        // Adjust confidence based on feature clarity
        if (cpu == 0 && memory == 0) {
            confidence *= 0.7; // Lower confidence with no resource requests
            factors.add("No resource requests specified");
        }

        // Determine workload type
        Map<String, String> annotations = pod.getMetadata().getAnnotations();
        WorkloadType type = determineWorkloadType(annotations, features);

        // Build reasoning
        reasoning.append("Category: ").append(category)
                 .append(", Type: ").append(type)
                 .append(", Confidence: ").append(String.format("%.2f", confidence))
                 .append("\nFactors: ").append(String.join("; ", factors));

        logger.info(String.format("Classified pod %s as %s (%s) with %.2f confidence",
            pod.getMetadata().getName(), category, type, confidence));

        return new ClassificationResult(category, type, confidence, features, reasoning.toString());
    }

    /**
     * Classify based on resource requirements
     */
    private static WorkloadCategory classifyByResources(double cpu, long memory, int linesOfCode,
                                                       boolean isDistributed, int containerCount,
                                                       boolean hasStatefulVolume, List<String> factors) {
        // DISTRIBUTED: High resources or complex setup
        if (cpu > SCALABLE_CPU_THRESHOLD || memory > SCALABLE_MEMORY_THRESHOLD || 
            linesOfCode > SCALABLE_LOC_THRESHOLD || hasStatefulVolume) {
            
            if (cpu > SCALABLE_CPU_THRESHOLD) 
                factors.add(String.format("High CPU: %.2f cores", cpu));
            if (memory > SCALABLE_MEMORY_THRESHOLD) 
                factors.add(String.format("High memory: %d MB", memory / (1024 * 1024)));
            if (linesOfCode > SCALABLE_LOC_THRESHOLD) 
                factors.add(String.format("Large codebase: %d LOC", linesOfCode));
            if (hasStatefulVolume) 
                factors.add("Stateful with persistent volumes");
            
            return WorkloadCategory.DISTRIBUTED;
        }
        
        // SCALABLE: Medium resources
        if (cpu > LIGHT_CPU_THRESHOLD || memory > LIGHT_MEMORY_THRESHOLD || 
            linesOfCode > LIGHT_LOC_THRESHOLD) {
            
            if (cpu > LIGHT_CPU_THRESHOLD) 
                factors.add(String.format("Medium CPU: %.2f cores", cpu));
            if (memory > LIGHT_MEMORY_THRESHOLD) 
                factors.add(String.format("Medium memory: %d MB", memory / (1024 * 1024)));
            if (linesOfCode > LIGHT_LOC_THRESHOLD) 
                factors.add(String.format("Medium codebase: %d LOC", linesOfCode));
            
            return WorkloadCategory.SCALABLE;
        }
        
        // LIGHT: Low resources
        factors.add(String.format("Light resources: %.2f CPU, %d MB memory", 
                                 cpu, memory / (1024 * 1024)));
        return WorkloadCategory.LIGHT;
    }

    /**
     * Determine workload type based on annotations and resource patterns
     */
    private static WorkloadType determineWorkloadType(Map<String, String> annotations, 
                                                      Map<String, Object> features) {
        // Check explicit annotation first
        if (annotations != null && annotations.containsKey("task.type")) {
            String taskType = annotations.get("task.type").toUpperCase().replace("_", "_");
            try {
                return WorkloadType.valueOf(taskType);
            } catch (IllegalArgumentException e) {
                logger.warning("Invalid task.type annotation: " + taskType);
            }
        }

        // Infer from resource patterns
        double cpu = (Double) features.get("cpu_cores");
        long memory = (Long) features.get("memory_bytes");
        double cpuMemoryRatio = (Double) features.get("cpu_memory_ratio");
        boolean isBatch = (Boolean) features.get("is_batch");

        // High CPU-to-memory ratio suggests compute-intensive
        if (cpuMemoryRatio > 0.5) {
            return WorkloadType.COMPUTE_INTENSIVE;
        }
        
        // High memory relative to CPU suggests memory-intensive
        if (memory > 2L * 1024 * 1024 * 1024 && cpuMemoryRatio < 0.3) {
            return WorkloadType.MEMORY_INTENSIVE;
        }

        // Batch jobs
        if (isBatch) {
            return WorkloadType.BATCH_PROCESSING;
        }

        // Low resources suggest energy-efficient workload
        if (cpu < 0.5 && memory < 512 * 1024 * 1024L) {
            return WorkloadType.ENERGY_EFFICIENT;
        }

        return WorkloadType.DEFAULT;
    }

    /**
     * Parse Kubernetes quantity string to double value
     */
    private static double parseQuantity(Quantity quantity) {
        if (quantity == null) {
            return 0.0;
        }

        try {
            if (quantity.getFormat().equals("DecimalSI")) {
                return quantity.getNumber().doubleValue();
            }

            String value = quantity.toSuffixedString();
            if (value.endsWith("m")) {
                return Double.parseDouble(value.substring(0, value.length() - 1)) / 1000.0;
            }

            String numericPart = value.replaceAll("[^\\d.]", "");
            double number = Double.parseDouble(numericPart);

            if (value.endsWith("Ki")) {
                return number * 1024;
            } else if (value.endsWith("Mi")) {
                return number * 1024 * 1024;
            } else if (value.endsWith("Gi")) {
                return number * 1024 * 1024 * 1024;
            } else if (value.endsWith("Ti")) {
                return number * 1024 * 1024 * 1024 * 1024L;
            } else if (value.endsWith("K")) {
                return number * 1000;
            } else if (value.endsWith("M")) {
                return number * 1000 * 1000;
            } else if (value.endsWith("G")) {
                return number * 1000 * 1000 * 1000;
            } else if (value.endsWith("T")) {
                return number * 1000 * 1000 * 1000 * 1000L;
            }

            return number;
        } catch (Exception e) {
            logger.warning(String.format("Failed to parse quantity '%s': %s", 
                quantity.toSuffixedString(), e.getMessage()));
            return 0.0;
        }
    }

    /**
     * Update pod annotations with classification results
     */
    public static void annotateWithClassification(V1Pod pod, ClassificationResult result) {
        Map<String, String> annotations = pod.getMetadata().getAnnotations();
        if (annotations == null) {
            annotations = new HashMap<>();
            pod.getMetadata().setAnnotations(annotations);
        }

        annotations.put("workload.category.auto", result.getCategory().toString());
        annotations.put("workload.type.auto", result.getType().toString());
        annotations.put("workload.classification.confidence", 
                       String.format("%.2f", result.getConfidence()));
        annotations.put("workload.classification.reasoning", result.getReasoning());
    }
}