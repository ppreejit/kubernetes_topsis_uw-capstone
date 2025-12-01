package com.scheduler;

import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.openapi.models.*;
import io.kubernetes.client.custom.V1Patch;
import com.scheduler.WorkloadClassifier.*;
import java.util.logging.Logger;

/**
 * Integration module for incorporating WorkloadClassifier into 
 * EnhancedMultiCriteriaScheduler
 */
public class SchedulerIntegration {
    private static final Logger logger = Logger.getLogger(SchedulerIntegration.class.getName());

    /**
     * Classify pod before scheduling and adjust weights accordingly
     */
    public static ClassificationResult classifyAndPreparePod(V1Pod pod) {
        ClassificationResult result = WorkloadClassifier.classifyWorkload(pod);
        
        logger.info(String.format(
            "Pod %s classified as %s (%s) with %.2f confidence",
            pod.getMetadata().getName(),
            result.getCategory(),
            result.getType(),
            result.getConfidence()
        ));
        
        WorkloadClassifier.annotateWithClassification(pod, result);
        return result;
    }

    /**
     * Adjust scheduling algorithm weights based on workload classification
     */
    public static double[] getAdjustedWeights(ClassificationResult classification) {
        double[] weights = {0.2, 0.2, 0.2, 0.2, 0.2}; // [execTime, energy, cpu, memory, balance]
        
        WorkloadCategory category = classification.getCategory();
        WorkloadType type = classification.getType();
        
        // Adjust based on category
        switch (category) {
            case LIGHT:
                weights = new double[]{0.15, 0.35, 0.15, 0.15, 0.20}; // Energy-focused
                break;
            case SCALABLE:
                weights = new double[]{0.25, 0.20, 0.20, 0.20, 0.15}; // Balanced
                break;
            case DISTRIBUTED:
                weights = new double[]{0.30, 0.10, 0.25, 0.25, 0.10}; // Performance-focused
                break;
        }
        
        // Further adjust based on type
        switch (type) {
            case COMPUTE_INTENSIVE:
                weights[2] *= 1.5; // CPU
                weights[0] *= 1.3; // Execution time
                normalizeWeights(weights);
                break;
            case MEMORY_INTENSIVE:
                weights[3] *= 1.5; // Memory
                normalizeWeights(weights);
                break;
            case ENERGY_EFFICIENT:
                weights[1] *= 1.8; // Energy
                normalizeWeights(weights);
                break;
        }
        
        return weights;
    }

    private static void normalizeWeights(double[] weights) {
        double sum = 0;
        for (double w : weights) sum += w;
        if (sum > 0) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] /= sum;
            }
        }
    }
}