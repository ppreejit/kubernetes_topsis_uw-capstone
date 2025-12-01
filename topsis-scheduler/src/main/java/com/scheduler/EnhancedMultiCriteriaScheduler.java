package com.scheduler;

import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.openapi.models.*;
import io.kubernetes.client.util.Config;
import io.kubernetes.client.custom.Quantity;
import io.kubernetes.client.custom.V1Patch;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.openapi.models.CoreV1Event;
import io.kubernetes.client.openapi.models.V1EventSource;

import com.scheduler.WorkloadClassifier;
import com.scheduler.WorkloadClassifier.ClassificationResult;
import com.scheduler.WorkloadClassifier.WorkloadCategory;
import com.scheduler.WorkloadClassifier.WorkloadType;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.time.Instant;
import java.time.Duration;

/**
 * Enhanced Kubernetes Pod scheduler that implements multiple decision-making
 * algorithms: - TOPSIS (Technique for Order of Preference by Similarity to
 * Ideal Solution) - VIKOR (VlseKriterijumska Optimizacija I Kompromisno
 * Resenje) for compromise solutions - Weighted Sum for fast decisions during
 * high-load situations
 * 
 * The scheduler automatically selects the appropriate algorithm based on
 * cluster load and criteria conflict levels.
 */
public class EnhancedMultiCriteriaScheduler {
	private static final Logger logger = Logger.getLogger(EnhancedMultiCriteriaScheduler.class.getName());
	private static CoreV1Api api;

	// Thresholds for algorithm selection
	private static final int HIGH_LOAD_THRESHOLD = 50; // Number of pending pods
	private static final double CONFLICT_THRESHOLD = 0.85; // Criteria conflict level
	private static final double VIKOR_NU = 0.5; // VIKOR decision-making strategy weight

	/**
	 * Enum for scheduling algorithms
	 */
	public enum SchedulingAlgorithm {
		TOPSIS, VIKOR, WEIGHTED_SUM
	}

	private static class SchedulingMetrics {
		private final String podName;
		private final Duration totalTime;
		private final Duration algorithmTime;
		private final Duration bindingTime;
		private final double energyConsumedJoules;
		private final SchedulingAlgorithm algorithm;
		private final double score;
		// NEW: Add classification info
		private final WorkloadCategory category;
		private final WorkloadType type;

		public SchedulingMetrics(String podName, Duration totalTime, Duration algorithmTime, Duration bindingTime,
				double energyConsumedJoules, SchedulingAlgorithm algorithm, double score, WorkloadCategory category,
				WorkloadType type) {
			this.podName = podName;
			this.totalTime = totalTime;
			this.algorithmTime = algorithmTime;
			this.bindingTime = bindingTime;
			this.energyConsumedJoules = energyConsumedJoules;
			this.algorithm = algorithm;
			this.score = score;
			this.category = category;
			this.type = type;
		}

		public void logMetrics() {
			double energyKilojoules = energyConsumedJoules / 1000.0;
			logger.info(String.format("Pod %s scheduling metrics:", podName));
			logger.info(String.format("- Workload category: %s", category));
			logger.info(String.format("- Workload type: %s", type));
			logger.info(String.format("- Algorithm used: %s", algorithm));
			logger.info(String.format("- Algorithm score: %.4f", score));
			logger.info(String.format("- Total scheduling time: %d ms", totalTime.toMillis()));
			logger.info(String.format("- Algorithm calculation time: %d ms", algorithmTime.toMillis()));
			logger.info(String.format("- Binding time: %d ms", bindingTime.toMillis()));
			logger.info(String.format("- Energy consumption: %.4f kJ", energyKilojoules));
		}

		public WorkloadCategory getCategory() {
			return category;
		}

		public WorkloadType getType() {
			return type;
		}
	}

	/**
	 * Base class for scheduling algorithm results
	 */
	private static abstract class SchedulingResult {
		protected final String selectedNode;
		protected final double score;
		protected final double estimatedEnergy;
		protected final SchedulingAlgorithm algorithm;

		public SchedulingResult(String selectedNode, double score, double estimatedEnergy,
				SchedulingAlgorithm algorithm) {
			this.selectedNode = selectedNode;
			this.score = score;
			this.estimatedEnergy = estimatedEnergy;
			this.algorithm = algorithm;
		}

		public String getSelectedNode() {
			return selectedNode;
		}

		public double getScore() {
			return score;
		}

		public double getEstimatedEnergy() {
			return estimatedEnergy;
		}

		public SchedulingAlgorithm getAlgorithm() {
			return algorithm;
		}
	}

	/**
	 * TOPSIS algorithm result
	 */
	private static class TopsisResult extends SchedulingResult {
		public TopsisResult(String selectedNode, double relativeCloseness, double estimatedEnergy) {
			super(selectedNode, relativeCloseness, estimatedEnergy, SchedulingAlgorithm.TOPSIS);
		}
	}

	/**
	 * VIKOR algorithm result
	 */
	private static class VikorResult extends SchedulingResult {
		private final double qValue;
		private final boolean isCompromise;

		public VikorResult(String selectedNode, double qValue, double estimatedEnergy, boolean isCompromise) {
			super(selectedNode, qValue, estimatedEnergy, SchedulingAlgorithm.VIKOR);
			this.qValue = qValue;
			this.isCompromise = isCompromise;
		}

		public double getQValue() {
			return qValue;
		}

		public boolean isCompromise() {
			return isCompromise;
		}
	}

	/**
	 * Weighted Sum algorithm result
	 */
	private static class WeightedSumResult extends SchedulingResult {
		public WeightedSumResult(String selectedNode, double weightedScore, double estimatedEnergy) {
			super(selectedNode, weightedScore, estimatedEnergy, SchedulingAlgorithm.WEIGHTED_SUM);
		}
	}

	public static void main(String[] args) throws IOException {
		try {
			// Load logging configuration
			InputStream loggingConfigStream = EnhancedMultiCriteriaScheduler.class.getClassLoader()
					.getResourceAsStream("logging.properties");
			if (loggingConfigStream != null) {
				LogManager.getLogManager().readConfiguration(loggingConfigStream);
				logger.info("Logging configuration loaded successfully");
			} else {
				logger.warning("Could not find logging.properties file");
			}

			logger.info("Initializing Enhanced Multi-Criteria Scheduler with Workload Classification...");

			// Initialize Kubernetes API client
			ApiClient client = ClientBuilder.cluster().build();
			client.setConnectTimeout(60000);
			client.setReadTimeout(60000);
			api = new CoreV1Api(client);

			logger.info("Kubernetes client initialized successfully");

			// Main scheduling loop
			while (true) {
				try {
					logger.info("Starting scheduling cycle");
					List<V1Pod> unscheduledPods = getUnscheduledPods(api);

					if (!unscheduledPods.isEmpty()) {
						logger.info(String.format("Found %d unscheduled pods", unscheduledPods.size()));
						List<V1Node> nodes = getAvailableNodes(api);

						if (nodes.isEmpty()) {
							logger.warning("No available nodes found in the cluster");
							continue;
						}

						logger.info(String.format("Found %d available nodes for scheduling", nodes.size()));

						// Process each unscheduled pod
						for (V1Pod pod : unscheduledPods) {
							try {
								Instant schedulingStart = Instant.now();

								// NEW: Classify workload before scheduling
								ClassificationResult classification = WorkloadClassifier.classifyWorkload(pod);
								logger.info(String.format("Pod %s classified as %s (%s) with %.2f confidence",
										pod.getMetadata().getName(), classification.getCategory(),
										classification.getType(), classification.getConfidence()));

								// NEW: Get adjusted weights based on classification
								double[] adjustedWeights = getAdjustedWeightsForClassification(classification);

								// Determine scheduling algorithm (now considers classification)
								SchedulingAlgorithm algorithm = selectSchedulingAlgorithm(unscheduledPods, nodes,
										classification);

								logger.info(String.format("Starting scheduling process for pod: %s using %s at %s",
										pod.getMetadata().getName(), algorithm, schedulingStart));

								// Run selected algorithm with adjusted weights
								SchedulingResult result = executeSchedulingAlgorithm(algorithm, nodes, pod,
										adjustedWeights);
								String bestNode = result.getSelectedNode();
								double energyEstimate = result.getEstimatedEnergy();

								Instant bindingStart = Instant.now();
								Duration algorithmTime = Duration.between(schedulingStart, bindingStart);
								logger.info(String.format("%s calculation completed for pod %s in %d ms", algorithm,
										pod.getMetadata().getName(), algorithmTime.toMillis()));

								bindPodToNode(api, pod, bestNode);

								Instant schedulingEnd = Instant.now();
								Duration totalTime = Duration.between(schedulingStart, schedulingEnd);
								Duration bindingTime = Duration.between(bindingStart, schedulingEnd);

								// Log metrics with classification info
								SchedulingMetrics metrics = new SchedulingMetrics(pod.getMetadata().getName(),
										totalTime, algorithmTime, bindingTime, energyEstimate, algorithm,
										result.getScore(), classification.getCategory(), classification.getType());
								metrics.logMetrics();

								// Store metrics in pod annotations
								storeSchedulingMetrics(pod, metrics, classification);

							} catch (Exception e) {
								logger.log(Level.SEVERE, String.format("Failed to schedule pod %s: %s",
										pod.getMetadata().getName(), e.getMessage()), e);
							}
						}
					} else {
						logger.fine("No unscheduled pods found");
					}

					logger.fine("Sleeping for 10 seconds before next scheduling cycle");
					Thread.sleep(10000);

				} catch (InterruptedException e) {
					logger.warning("Scheduler interrupted: " + e.getMessage());
					Thread.currentThread().interrupt();
					break;
				} catch (ApiException e) {
					logger.severe("Kubernetes API error: " + e.getMessage());
					logger.severe("Response Body: " + e.getResponseBody());
					Thread.sleep(30000);
				} catch (Exception e) {
					logger.severe("Unexpected error: " + e.getMessage());
					Thread.sleep(30000);
				}
			}
		} catch (Exception e) {
			logger.severe("Fatal error in scheduler: " + e.getMessage());
			System.exit(1);
		}
	}

	/**
	 * Get adjusted weights based on workload classification
	 */
	private static double[] getAdjustedWeightsForClassification(ClassificationResult classification) {
		// Base weights: [execTime, energy, cpu, memory, balance]
		double[] weights = { 0.2, 0.2, 0.2, 0.2, 0.2 };

		WorkloadCategory category = classification.getCategory();
		WorkloadType type = classification.getType();

		// Adjust based on category
		switch (category) {
		case LIGHT:
			// Light workloads: prioritize energy efficiency
			weights[1] = 0.35; // energy
			weights[0] = 0.15; // execution time
			weights[2] = 0.15; // cpu
			weights[3] = 0.15; // memory
			weights[4] = 0.20; // balance
			logger.info("Applied LIGHT workload weights (energy-focused)");
			break;

		case SCALABLE:
			// Scalable workloads: balanced approach
			weights[0] = 0.25; // execution time
			weights[1] = 0.20; // energy
			weights[2] = 0.20; // cpu
			weights[3] = 0.20; // memory
			weights[4] = 0.15; // balance
			logger.info("Applied SCALABLE workload weights (balanced)");
			break;

		case DISTRIBUTED:
			// Distributed workloads: prioritize performance and resources
			weights[0] = 0.30; // execution time
			weights[1] = 0.10; // energy (less important)
			weights[2] = 0.25; // cpu
			weights[3] = 0.25; // memory
			weights[4] = 0.10; // balance
			logger.info("Applied DISTRIBUTED workload weights (performance-focused)");
			break;
		}

		// Further adjust based on workload type
		switch (type) {
		case COMPUTE_INTENSIVE:
			weights[2] *= 1.5; // increase CPU importance
			weights[0] *= 1.3; // increase execution time importance
			normalizeWeights(weights);
			logger.info("Adjusted for COMPUTE_INTENSIVE type");
			break;

		case MEMORY_INTENSIVE:
			weights[3] *= 1.5; // increase memory importance
			normalizeWeights(weights);
			logger.info("Adjusted for MEMORY_INTENSIVE type");
			break;

		case ENERGY_EFFICIENT:
			weights[1] *= 1.8; // significantly increase energy importance
			normalizeWeights(weights);
			logger.info("Adjusted for ENERGY_EFFICIENT type");
			break;

		case BATCH_PROCESSING:
			weights[0] *= 0.7; // decrease execution time importance
			weights[1] *= 1.4; // increase energy importance
			normalizeWeights(weights);
			logger.info("Adjusted for BATCH_PROCESSING type");
			break;

		case REAL_TIME:
			weights[0] *= 1.8; // significantly increase execution time importance
			weights[4] *= 1.3; // increase balance for stability
			normalizeWeights(weights);
			logger.info("Adjusted for REAL_TIME type");
			break;

		case IO_INTENSIVE:
		case DEFAULT:
			// No additional adjustments
			break;
		}

		logger.info(String.format("Final weights: execTime=%.3f, energy=%.3f, cpu=%.3f, memory=%.3f, balance=%.3f",
				weights[0], weights[1], weights[2], weights[3], weights[4]));

		return weights;
	}

	/**
	 * Executes the selected scheduling algorithm with custom weights
	 */
	private static SchedulingResult executeSchedulingAlgorithm(SchedulingAlgorithm algorithm, List<V1Node> nodes,
			V1Pod pod, double[] weights) throws ApiException {
		switch (algorithm) {
		case TOPSIS:
			return topsisSchedule(nodes, pod, weights);
		case VIKOR:
			return vikorSchedule(nodes, pod, weights);
		case WEIGHTED_SUM:
			return weightedSumSchedule(nodes, pod, weights);
		default:
			throw new IllegalArgumentException("Unknown scheduling algorithm: " + algorithm);
		}
	}

	/**
	 * Calculates the level of conflict between criteria by analyzing the
	 * correlation between different metrics across nodes
	 */
	private static double calculateCriteriaConflict(List<V1Node> nodes, V1Pod samplePod) {
		if (nodes.size() < 2)
			return 0.0;

		try {
			Task task = new Task(samplePod);
			double[] executionTimes = new double[nodes.size()];
			double[] energyValues = new double[nodes.size()];
			double[] cpuAvailability = new double[nodes.size()];
			double[] memoryAvailability = new double[nodes.size()];

			// Calculate metrics for each node
			for (int i = 0; i < nodes.size(); i++) {
				NodeMetrics metrics = new NodeMetrics(nodes.get(i));
				executionTimes[i] = metrics.estimateExecutionTime(task);
				energyValues[i] = metrics.estimateEnergy(task);
				cpuAvailability[i] = metrics.availableCores;
				memoryAvailability[i] = metrics.availableMemory;
			}

			// Calculate correlations between criteria (negative correlation indicates
			// conflict)
			double execEnergyCorr = Math.abs(calculateCorrelation(executionTimes, energyValues));
			double execCpuCorr = Math.abs(calculateCorrelation(executionTimes, cpuAvailability));
			double energyCpuCorr = Math.abs(calculateCorrelation(energyValues, cpuAvailability));
			double cpuMemCorr = Math.abs(calculateCorrelation(cpuAvailability, memoryAvailability));

			// Average correlation - lower correlation means higher conflict
			double avgCorrelation = (execEnergyCorr + execCpuCorr + energyCpuCorr + cpuMemCorr) / 4.0;
			return 1.0 - avgCorrelation; // Convert to conflict level

		} catch (Exception e) {
			logger.warning("Error calculating criteria conflict: " + e.getMessage());
			return 0.0;
		}
	}

	/**
	 * Calculates Pearson correlation coefficient between two arrays
	 */
	private static double calculateCorrelation(double[] x, double[] y) {
		if (x.length != y.length || x.length == 0)
			return 0.0;

		double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
		int n = x.length;

		for (int i = 0; i < n; i++) {
			sumX += x[i];
			sumY += y[i];
			sumXY += x[i] * y[i];
			sumX2 += x[i] * x[i];
			sumY2 += y[i] * y[i];
		}

		double numerator = n * sumXY - sumX * sumY;
		double denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

		return denominator == 0 ? 0 : numerator / denominator;
	}

	/**
	 * Selects scheduling algorithm based on cluster state AND workload
	 * classification
	 */
	private static SchedulingAlgorithm selectSchedulingAlgorithm(List<V1Pod> unscheduledPods, List<V1Node> nodes,
			ClassificationResult classification) {

		WorkloadCategory category = classification.getCategory();
		WorkloadType type = classification.getType();

		// High load scenario - use fast Weighted Sum
		if (unscheduledPods.size() >= HIGH_LOAD_THRESHOLD) {
			logger.info(String.format("High load detected (%d pending pods), using Weighted Sum for speed",
					unscheduledPods.size()));
			return SchedulingAlgorithm.WEIGHTED_SUM;
		}

		// LIGHT workloads with high confidence can use fast Weighted Sum
		if (category == WorkloadCategory.LIGHT && classification.getConfidence() > 0.85) {
			logger.info("LIGHT workload with high confidence - using Weighted Sum for efficiency");
			return SchedulingAlgorithm.WEIGHTED_SUM;
		}

		// REAL_TIME workloads should use fast algorithm
		if (type == WorkloadType.REAL_TIME) {
			logger.info("REAL_TIME workload - using Weighted Sum for speed");
			return SchedulingAlgorithm.WEIGHTED_SUM;
		}

		// Calculate criteria conflict level to decide between TOPSIS and VIKOR
		double conflictLevel = calculateCriteriaConflict(nodes, unscheduledPods.get(0));

		// DISTRIBUTED workloads with conflicting requirements benefit from VIKOR
		if (category == WorkloadCategory.DISTRIBUTED && conflictLevel >= 0.7) {
			logger.info("DISTRIBUTED workload with high conflict - using VIKOR for compromise");
			return SchedulingAlgorithm.VIKOR;
		}

		if (conflictLevel >= CONFLICT_THRESHOLD) {
			logger.info(String.format("High criteria conflict detected (%.3f), using VIKOR for compromise solution",
					conflictLevel));
			return SchedulingAlgorithm.VIKOR;
		} else {
			logger.info(String.format("Normal conflict level (%.3f), using TOPSIS", conflictLevel));
			return SchedulingAlgorithm.TOPSIS;
		}
	}

	/**
	 * VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje) algorithm
	 * implementation Finds compromise solutions when criteria are conflicting
	 */
	private static VikorResult vikorSchedule(List<V1Node> nodes, V1Pod pod) throws ApiException {
		Instant start = Instant.now();

		if (nodes.isEmpty()) {
			throw new IllegalArgumentException("No nodes available for scheduling");
		}

		List<V1Node> eligibleNodes = getEligibleNodes(nodes, pod);
		if (eligibleNodes.isEmpty()) {
			throw new ApiException("No nodes have sufficient resources for pod " + pod.getMetadata().getName());
		}

		Task task = new Task(pod);
		logger.info(
				String.format("Starting VIKOR calculation for task %s on %d nodes", task.name, eligibleNodes.size()));

		int numCriteria = 5;
		double[] weights = { 0.2, 0.2, 0.2, 0.2, 0.2 };
		double[][] criteriaMatrix = new double[eligibleNodes.size()][numCriteria];
		double[] nodeEnergyEstimates = new double[eligibleNodes.size()];

		// Step 1: Build criteria matrix
		for (int i = 0; i < eligibleNodes.size(); i++) {
			try {
				NodeMetrics metrics = new NodeMetrics(eligibleNodes.get(i));
				double executionTime = metrics.estimateExecutionTime(task);
				double energy = metrics.estimateEnergy(task);
				nodeEnergyEstimates[i] = energy;
				double resourceBalance = Math.abs(
						metrics.availableCores / metrics.totalCores - metrics.availableMemory / metrics.totalMemory);

				// For VIKOR, use raw values (will be normalized relative to best/worst)
				criteriaMatrix[i][0] = executionTime;
				criteriaMatrix[i][1] = energy;
				criteriaMatrix[i][2] = metrics.availableCores;
				criteriaMatrix[i][3] = metrics.availableMemory;
				criteriaMatrix[i][4] = resourceBalance;

			} catch (Exception e) {
				logger.warning(String.format("Error calculating metrics for node %s: %s",
						eligibleNodes.get(i).getMetadata().getName(), e.getMessage()));
				Arrays.fill(criteriaMatrix[i], 0.0);
			}
		}

		// Step 2: Find best and worst values for each criterion
		double[] bestValues = new double[numCriteria];
		double[] worstValues = new double[numCriteria];

		for (int j = 0; j < numCriteria; j++) {
			bestValues[j] = criteriaMatrix[0][j];
			worstValues[j] = criteriaMatrix[0][j];

			for (int i = 1; i < eligibleNodes.size(); i++) {
				// For minimization criteria (0,1,4): best = min, worst = max
				// For maximization criteria (2,3): best = max, worst = min
				if (j == 0 || j == 1 || j == 4) {
					bestValues[j] = Math.min(bestValues[j], criteriaMatrix[i][j]);
					worstValues[j] = Math.max(worstValues[j], criteriaMatrix[i][j]);
				} else {
					bestValues[j] = Math.max(bestValues[j], criteriaMatrix[i][j]);
					worstValues[j] = Math.min(worstValues[j], criteriaMatrix[i][j]);
				}
			}
		}

		// Step 3: Calculate S and R values for each alternative
		double[] sValues = new double[eligibleNodes.size()]; // Utility measure
		double[] rValues = new double[eligibleNodes.size()]; // Regret measure

		for (int i = 0; i < eligibleNodes.size(); i++) {
			double sSum = 0.0;
			double rMax = 0.0;

			for (int j = 0; j < numCriteria; j++) {
				double range = worstValues[j] - bestValues[j];
				double normalizedValue = range == 0 ? 0 : (Math.abs(bestValues[j] - criteriaMatrix[i][j])) / range;

				double weightedValue = weights[j] * normalizedValue;
				sSum += weightedValue;
				rMax = Math.max(rMax, weightedValue);
			}

			sValues[i] = sSum;
			rValues[i] = rMax;
		}

		// Step 4: Calculate Q values (compromise measure)
		double sBest = Arrays.stream(sValues).min().orElse(0);
		double sWorst = Arrays.stream(sValues).max().orElse(1);
		double rBest = Arrays.stream(rValues).min().orElse(0);
		double rWorst = Arrays.stream(rValues).max().orElse(1);

		double[] qValues = new double[eligibleNodes.size()];
		for (int i = 0; i < eligibleNodes.size(); i++) {
			double sNorm = (sWorst - sBest) == 0 ? 0 : (sValues[i] - sBest) / (sWorst - sBest);
			double rNorm = (rWorst - rBest) == 0 ? 0 : (rValues[i] - rBest) / (rWorst - rBest);
			qValues[i] = VIKOR_NU * sNorm + (1 - VIKOR_NU) * rNorm;
		}

		// Step 5: Find the best alternative and check compromise conditions
		int bestIndex = 0;
		double bestQ = qValues[0];
		for (int i = 1; i < eligibleNodes.size(); i++) {
			if (qValues[i] < bestQ) {
				bestQ = qValues[i];
				bestIndex = i;
			}
		}

		// Check compromise conditions
		boolean isAcceptableAdvantage = true;
		boolean isAcceptableStability = true;

		if (eligibleNodes.size() > 1) {
			// Find second best Q value
			double secondBestQ = Double.MAX_VALUE;
			for (int i = 0; i < eligibleNodes.size(); i++) {
				if (i != bestIndex && qValues[i] < secondBestQ) {
					secondBestQ = qValues[i];
				}
			}

			// Acceptable advantage condition: Q(A2) - Q(A1) >= DQ
			double dq = 1.0 / (eligibleNodes.size() - 1);
			isAcceptableAdvantage = (secondBestQ - bestQ) >= dq;

			// Acceptable stability: A1 must be best in S or R
			double bestS = Arrays.stream(sValues).min().orElse(0);
			double bestR = Arrays.stream(rValues).min().orElse(0);
			isAcceptableStability = (sValues[bestIndex] == bestS) || (rValues[bestIndex] == bestR);
		}

		boolean isCompromise = isAcceptableAdvantage && isAcceptableStability;
		String selectedNode = eligibleNodes.get(bestIndex).getMetadata().getName();
		double estimatedEnergy = nodeEnergyEstimates[bestIndex];

		Duration totalTime = Duration.between(start, Instant.now());
		logger.info(String.format("VIKOR calculation completed in %d ms", totalTime.toMillis()));
		logger.info(String.format("Selected node %s with Q-value %.4f (compromise: %s)", selectedNode, bestQ,
				isCompromise));
		logger.info(String.format("Estimated energy consumption: %.4f J (%.4f kJ)", estimatedEnergy,
				estimatedEnergy / 1000.0));

		return new VikorResult(selectedNode, bestQ, estimatedEnergy, isCompromise);
	}

	/**
	 * Weighted Sum algorithm implementation for fast scheduling decisions
	 */
	private static WeightedSumResult weightedSumSchedule(List<V1Node> nodes, V1Pod pod) throws ApiException {
		Instant start = Instant.now();

		if (nodes.isEmpty()) {
			throw new IllegalArgumentException("No nodes available for scheduling");
		}

		List<V1Node> eligibleNodes = getEligibleNodes(nodes, pod);
		if (eligibleNodes.isEmpty()) {
			throw new ApiException("No nodes have sufficient resources for pod " + pod.getMetadata().getName());
		}

		Task task = new Task(pod);
		logger.info(String.format("Starting Weighted Sum calculation for task %s on %d nodes", task.name,
				eligibleNodes.size()));

		// Adaptive weights based on task characteristics
		double[] weights = calculateAdaptiveWeights(task);
		double[] nodeEnergyEstimates = new double[eligibleNodes.size()];

		// Step 1: Calculate raw scores for each node
		double[] executionTimes = new double[eligibleNodes.size()];
		double[] energyValues = new double[eligibleNodes.size()];
		double[] cpuAvailability = new double[eligibleNodes.size()];
		double[] memoryAvailability = new double[eligibleNodes.size()];
		double[] resourceBalance = new double[eligibleNodes.size()];

		for (int i = 0; i < eligibleNodes.size(); i++) {
			try {
				NodeMetrics metrics = new NodeMetrics(eligibleNodes.get(i));
				executionTimes[i] = metrics.estimateExecutionTime(task);
				energyValues[i] = metrics.estimateEnergy(task);
				nodeEnergyEstimates[i] = energyValues[i];
				cpuAvailability[i] = metrics.availableCores;
				memoryAvailability[i] = metrics.availableMemory;
				resourceBalance[i] = Math.abs(
						metrics.availableCores / metrics.totalCores - metrics.availableMemory / metrics.totalMemory);
			} catch (Exception e) {
				logger.warning(String.format("Error calculating metrics for node %s: %s",
						eligibleNodes.get(i).getMetadata().getName(), e.getMessage()));
				// Set to worst possible values
				executionTimes[i] = Double.MAX_VALUE;
				energyValues[i] = Double.MAX_VALUE;
				cpuAvailability[i] = 0.0;
				memoryAvailability[i] = 0.0;
				resourceBalance[i] = 1.0;
			}
		}

		// Step 2: Normalize criteria (simple min-max normalization for speed)
		normalizeArray(executionTimes, true); // minimize
		normalizeArray(energyValues, true); // minimize
		normalizeArray(cpuAvailability, false); // maximize
		normalizeArray(memoryAvailability, false); // maximize
		normalizeArray(resourceBalance, true); // minimize

		// Step 3: Calculate weighted sum for each node
		double bestScore = Double.MIN_VALUE;
		int bestIndex = 0;

		for (int i = 0; i < eligibleNodes.size(); i++) {
			double score = weights[0] * executionTimes[i] + weights[1] * energyValues[i]
					+ weights[2] * cpuAvailability[i] + weights[3] * memoryAvailability[i]
					+ weights[4] * resourceBalance[i];

			if (score > bestScore) {
				bestScore = score;
				bestIndex = i;
			}
		}

		String selectedNode = eligibleNodes.get(bestIndex).getMetadata().getName();
		double estimatedEnergy = nodeEnergyEstimates[bestIndex];

		Duration totalTime = Duration.between(start, Instant.now());
		logger.info(String.format("Weighted Sum calculation completed in %d ms", totalTime.toMillis()));
		logger.info(String.format("Selected node %s with weighted score %.4f", selectedNode, bestScore));
		logger.info(String.format("Estimated energy consumption: %.4f J (%.4f kJ)", estimatedEnergy,
				estimatedEnergy / 1000.0));

		return new WeightedSumResult(selectedNode, bestScore, estimatedEnergy);
	}

	/**
	 * Calculates adaptive weights based on task characteristics
	 */
	private static double[] calculateAdaptiveWeights(Task task) {
		Map<String, String> annotations = task.annotations;
		String workloadType = annotations.getOrDefault("task.type", "default");
		String priority = annotations.getOrDefault("task.priority", "normal");

		// Base weights
		double[] weights = { 0.2, 0.2, 0.2, 0.2, 0.2 }; // exec, energy, cpu, memory, balance

		// Adjust weights based on workload type
		switch (workloadType.toLowerCase()) {
		case "compute_intensive":
			weights[0] = 0.35; // Higher weight on execution time
			weights[2] = 0.35; // Higher weight on CPU availability
			weights[1] = 0.15; // Lower weight on energy
			weights[3] = 0.1; // Lower weight on memory
			weights[4] = 0.05; // Lower weight on balance
			break;
		case "memory_intensive":
			weights[3] = 0.4; // Higher weight on memory availability
			weights[0] = 0.25; // Medium weight on execution time
			weights[1] = 0.15; // Lower weight on energy
			weights[2] = 0.15; // Lower weight on CPU
			weights[4] = 0.05; // Lower weight on balance
			break;
		case "energy_efficient":
			weights[1] = 0.5; // Much higher weight on energy
			weights[0] = 0.2; // Lower weight on execution time
			weights[2] = 0.1; // Lower weight on CPU
			weights[3] = 0.1; // Lower weight on memory
			weights[4] = 0.1; // Lower weight on balance
			break;
		}

		// Adjust for priority
		if ("high".equals(priority)) {
			// High priority tasks prioritize performance over energy
			weights[0] *= 1.5; // Increase execution time weight
			weights[1] *= 0.5; // Decrease energy weight
			normalizeWeights(weights);
		} else if ("low".equals(priority)) {
			// Low priority tasks can tolerate slower execution for better energy efficiency
			weights[0] *= 0.7; // Decrease execution time weight
			weights[1] *= 1.5; // Increase energy weight
			normalizeWeights(weights);
		}

		return weights;
	}

	/**
	 * Normalizes weights to sum to 1.0
	 */
	private static void normalizeWeights(double[] weights) {
		double sum = Arrays.stream(weights).sum();
		if (sum > 0) {
			for (int i = 0; i < weights.length; i++) {
				weights[i] /= sum;
			}
		}
	}

	/**
	 * Normalizes an array using min-max normalization
	 */
	private static void normalizeArray(double[] array, boolean minimize) {
		double min = Arrays.stream(array).min().orElse(0);
		double max = Arrays.stream(array).max().orElse(1);
		double range = max - min;

		if (range == 0) {
			Arrays.fill(array, 0.5); // All values are the same
			return;
		}

		for (int i = 0; i < array.length; i++) {
			if (minimize) {
				array[i] = (max - array[i]) / range; // Higher score for lower values
			} else {
				array[i] = (array[i] - min) / range; // Higher score for higher values
			}
		}
	}

	/**
	 * Original TOPSIS implementation (updated for new API)
	 */
	private static TopsisResult topsisSchedule(List<V1Node> nodes, V1Pod pod) throws ApiException {
		Instant start = Instant.now();

		if (nodes.isEmpty()) {
			throw new IllegalArgumentException("No nodes available for scheduling");
		}

		List<V1Node> eligibleNodes = getEligibleNodes(nodes, pod);
		if (eligibleNodes.isEmpty()) {
			throw new ApiException("No nodes have sufficient resources for pod " + pod.getMetadata().getName()
					+ String.format(" (Required: CPU=%.3f, Memory=%d)", getResourceRequest(pod, "cpu"),
							(long) getResourceRequest(pod, "memory")));
		}

		Task task = new Task(pod);
		logger.info(
				String.format("Starting TOPSIS calculation for task %s on %d nodes", task.name, eligibleNodes.size()));

		int numCriteria = 5;
		double[] weights = { 0.2, 0.2, 0.2, 0.2, 0.2 };
		double[][] decisionMatrix = new double[eligibleNodes.size()][numCriteria];
		double[] nodeEnergyEstimates = new double[eligibleNodes.size()];

		// Step 1: Create decision matrix using eligible nodes
		for (int i = 0; i < eligibleNodes.size(); i++) {
			try {
				NodeMetrics metrics = new NodeMetrics(eligibleNodes.get(i));
				double executionTime = metrics.estimateExecutionTime(task);
				double energy = metrics.estimateEnergy(task);
				nodeEnergyEstimates[i] = energy;
				double resourceBalance = Math.abs(
						metrics.availableCores / metrics.totalCores - metrics.availableMemory / metrics.totalMemory);

				decisionMatrix[i][0] = -executionTime; // Minimize execution time
				decisionMatrix[i][1] = -energy; // Minimize energy
				decisionMatrix[i][2] = metrics.availableCores; // Maximize core availability
				decisionMatrix[i][3] = metrics.availableMemory;// Maximize memory availability
				decisionMatrix[i][4] = -resourceBalance; // Minimize resource imbalance

			} catch (Exception e) {
				logger.warning(String.format("Error calculating metrics for node %s: %s",
						eligibleNodes.get(i).getMetadata().getName(), e.getMessage()));
				Arrays.fill(decisionMatrix[i], 0.0);
			}
		}

		// Step 2: Normalize the decision matrix
		double[] columnSums = new double[numCriteria];
		for (int j = 0; j < numCriteria; j++) {
			for (int i = 0; i < eligibleNodes.size(); i++) {
				columnSums[j] += Math.pow(decisionMatrix[i][j], 2);
			}
			columnSums[j] = Math.sqrt(columnSums[j]);
			if (columnSums[j] == 0) {
				columnSums[j] = 1;
			}
		}

		for (int i = 0; i < eligibleNodes.size(); i++) {
			for (int j = 0; j < numCriteria; j++) {
				decisionMatrix[i][j] = (decisionMatrix[i][j] / columnSums[j]) * weights[j];
			}
		}

		// Step 3: Calculate ideal and negative-ideal solutions
		double[] idealSolution = new double[numCriteria];
		double[] negativeIdealSolution = new double[numCriteria];
		Arrays.fill(idealSolution, Double.MIN_VALUE);
		Arrays.fill(negativeIdealSolution, Double.MAX_VALUE);

		for (int j = 0; j < numCriteria; j++) {
			for (int i = 0; i < eligibleNodes.size(); i++) {
				idealSolution[j] = Math.max(idealSolution[j], decisionMatrix[i][j]);
				negativeIdealSolution[j] = Math.min(negativeIdealSolution[j], decisionMatrix[i][j]);
			}
		}

		// Step 4: Calculate separations
		double[] separationIdeal = new double[eligibleNodes.size()];
		double[] separationNegativeIdeal = new double[eligibleNodes.size()];

		for (int i = 0; i < eligibleNodes.size(); i++) {
			for (int j = 0; j < numCriteria; j++) {
				separationIdeal[i] += Math.pow(decisionMatrix[i][j] - idealSolution[j], 2);
				separationNegativeIdeal[i] += Math.pow(decisionMatrix[i][j] - negativeIdealSolution[j], 2);
			}
			separationIdeal[i] = Math.sqrt(separationIdeal[i]);
			separationNegativeIdeal[i] = Math.sqrt(separationNegativeIdeal[i]);
		}

		// Step 5: Calculate relative closeness and select best node
		double maxCloseness = Double.MIN_VALUE;
		int bestNodeIndex = 0;

		for (int i = 0; i < eligibleNodes.size(); i++) {
			double separationSum = separationIdeal[i] + separationNegativeIdeal[i];
			double relativeCloseness = separationSum == 0 ? 0 : separationNegativeIdeal[i] / separationSum;

			if (relativeCloseness > maxCloseness) {
				maxCloseness = relativeCloseness;
				bestNodeIndex = i;
			}
		}

		String selectedNode = eligibleNodes.get(bestNodeIndex).getMetadata().getName();
		double estimatedEnergy = nodeEnergyEstimates[bestNodeIndex];
		Duration totalTime = Duration.between(start, Instant.now());

		logger.info(String.format("TOPSIS calculation completed in %d ms", totalTime.toMillis()));
		logger.info(String.format("Selected node %s with relative closeness %f", selectedNode, maxCloseness));
		logger.info(String.format("Estimated energy consumption: %.4f J (%.4f kJ)", estimatedEnergy,
				estimatedEnergy / 1000.0));

		return new TopsisResult(selectedNode, maxCloseness, estimatedEnergy);
	}

	/**
	 * Stores scheduling metrics as annotations on the pod with algorithm
	 * information
	 */
	private static void storeSchedulingMetrics(V1Pod pod, SchedulingMetrics metrics,
			ClassificationResult classification) {
		try {
			V1Patch patch = new V1Patch(String.format(
					"{\"metadata\":{\"annotations\":{" + "\"scheduler.algorithm\":\"%s\","
							+ "\"scheduler.score\":\"%.4f\"," + "\"scheduler.metrics.totalTimeMs\":\"%d\","
							+ "\"scheduler.metrics.algorithmTimeMs\":\"%d\","
							+ "\"scheduler.metrics.bindingTimeMs\":\"%d\"," + "\"scheduler.metrics.energyKJ\":\"%.4f\","
							+ "\"workload.category.auto\":\"%s\"," + "\"workload.type.auto\":\"%s\","
							+ "\"workload.classification.confidence\":\"%.2f\"" + "}}}",
					metrics.algorithm.toString(), metrics.score, metrics.totalTime.toMillis(),
					metrics.algorithmTime.toMillis(), metrics.bindingTime.toMillis(),
					metrics.energyConsumedJoules / 1000.0, classification.getCategory(), classification.getType(),
					classification.getConfidence()));

			api.patchNamespacedPod(pod.getMetadata().getName(), pod.getMetadata().getNamespace(), patch, null, null,
					null, null, null);

			logger.info(String.format("Stored enhanced metrics for pod %s", pod.getMetadata().getName()));
		} catch (ApiException e) {
			logger.warning(String.format("Failed to store metrics: %s", e.getMessage()));
		}
	}

	// ============= UTILITY METHODS (FIXED FOR NEW API) =============

	/**
	 * Gets unscheduled pods - FIXED for Kubernetes client v19.0.0 API
	 */
	private static List<V1Pod> getUnscheduledPods(CoreV1Api api) throws ApiException {
		// Updated to use the correct 11-parameter signature
		V1PodList podList = api.listPodForAllNamespaces(null, // allowWatchBookmarks
				null, // _continue
				null, // fieldSelector
				null, // labelSelector
				null, // limit
				null, // pretty
				null, // resourceVersion
				null, // resourceVersionMatch
				null, // sendInitialEvents
				null, // timeoutSeconds
				null // watch
		);

		return podList.getItems().stream()
				.filter(pod -> pod.getSpec().getNodeName() == null && pod.getSpec().getSchedulerName() != null
						&& pod.getSpec().getSchedulerName().equals("enhanced-scheduler"))
				.collect(Collectors.toList());
	}

	/**
	 * Gets available nodes - FIXED for Kubernetes client v19.0.0 API
	 */
	private static List<V1Node> getAvailableNodes(CoreV1Api api) throws ApiException {
		// Updated to use the correct 11-parameter signature
		V1NodeList nodeList = api.listNode(null, // pretty
				null, // allowWatchBookmarks
				null, // _continue
				null, // fieldSelector
				null, // labelSelector
				null, // limit
				null, // resourceVersion
				null, // resourceVersionMatch
				null, // sendInitialEvents
				null, // timeoutSeconds
				null // watch
		);

		return nodeList.getItems().stream()
				.filter(node -> node.getStatus().getConditions().stream().anyMatch(
						condition -> condition.getType().equals("Ready") && condition.getStatus().equals("True")))
				.collect(Collectors.toList());
	}

	/**
	 * Gets pods by node name - FIXED for Kubernetes client v19.0.0 API
	 */
	private static List<V1Pod> getPodsByNodeName(String nodeName) throws ApiException {
		String fieldSelector = String.format("spec.nodeName=%s", nodeName);

		// Updated to use the correct 11-parameter signature
		V1PodList podList = api.listPodForAllNamespaces(null, // allowWatchBookmarks
				null, // _continue
				fieldSelector, // fieldSelector
				null, // labelSelector
				null, // limit
				null, // pretty
				null, // resourceVersion
				null, // resourceVersionMatch
				null, // sendInitialEvents
				null, // timeoutSeconds
				null // watch
		);

		return podList.getItems();
	}

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
				return number * 1024 * 1024 * 1024 * 1024;
			} else if (value.endsWith("K")) {
				return number * 1000;
			} else if (value.endsWith("M")) {
				return number * 1000 * 1000;
			} else if (value.endsWith("G")) {
				return number * 1000 * 1000 * 1000;
			} else if (value.endsWith("T")) {
				return number * 1000 * 1000 * 1000 * 1000;
			}

			return number;
		} catch (Exception e) {
			logger.warning(
					String.format("Failed to parse quantity '%s': %s", quantity.toSuffixedString(), e.getMessage()));
			return 0.0;
		}
	}

	private static List<V1Node> getEligibleNodes(List<V1Node> nodes, V1Pod pod) throws ApiException {
		List<V1Node> eligibleNodes = new ArrayList<>();
		double requiredCpu = getResourceRequest(pod, "cpu");
		double requiredMemoryBytes = getResourceRequest(pod, "memory");

		for (V1Node node : nodes) {
			try {
				Map<String, Quantity> allocatable = node.getStatus().getAllocatable();
				double nodeCpu = parseQuantity(allocatable.get("cpu"));
				double nodeMemoryBytes = parseQuantity(allocatable.get("memory"));

				List<V1Pod> nodePods = getPodsByNodeName(node.getMetadata().getName());
				double usedCpu = nodePods.stream().filter(p -> p.getStatus() != null
						&& ("Running".equals(p.getStatus().getPhase()) || "Pending".equals(p.getStatus().getPhase())))
						.mapToDouble(p -> getResourceRequest(p, "cpu")).sum();
				double usedMemoryBytes = nodePods.stream().filter(p -> p.getStatus() != null
						&& ("Running".equals(p.getStatus().getPhase()) || "Pending".equals(p.getStatus().getPhase())))
						.mapToDouble(p -> getResourceRequest(p, "memory")).sum();

				double availableCpu = nodeCpu - usedCpu;
				double availableMemoryBytes = nodeMemoryBytes - usedMemoryBytes;

				if (availableCpu >= requiredCpu && availableMemoryBytes >= requiredMemoryBytes) {
					eligibleNodes.add(node);
				}
			} catch (ApiException e) {
				logger.warning(String.format("Failed to check resources for node %s: %s", node.getMetadata().getName(),
						e.getMessage()));
			}
		}

		return eligibleNodes;
	}

	private static double getResourceRequest(V1Pod pod, String resourceName) {
		if (pod.getSpec() == null || pod.getSpec().getContainers() == null) {
			return 0.0;
		}

		return pod.getSpec().getContainers().stream()
				.filter(container -> container.getResources() != null && container.getResources().getRequests() != null
						&& container.getResources().getRequests().containsKey(resourceName))
				.mapToDouble(container -> {
					Quantity quantity = container.getResources().getRequests().get(resourceName);
					return parseQuantity(quantity);
				}).sum();
	}

	private static void bindPodToNode(CoreV1Api api, V1Pod pod, String nodeName) throws ApiException {
		try {
			V1Binding binding = new V1Binding().metadata(new V1ObjectMeta().name(pod.getMetadata().getName()))
					.target(new V1ObjectReference().apiVersion("v1").kind("Node").name(nodeName));

			api.createNamespacedPodBinding(pod.getMetadata().getName(), pod.getMetadata().getNamespace(), binding, null,
					null, null, null);

			CoreV1Event event = new CoreV1Event()
					.metadata(new V1ObjectMeta()
							.name("enhanced-scheduled-" + pod.getMetadata().getName() + "-"
									+ UUID.randomUUID().toString().substring(0, 8))
							.namespace(pod.getMetadata().getNamespace()))
					.type("Normal").reason("Scheduled").message("Pod scheduled by enhanced-multi-criteria-scheduler")
					.involvedObject(new V1ObjectReference().kind("Pod").name(pod.getMetadata().getName())
							.namespace(pod.getMetadata().getNamespace()).uid(pod.getMetadata().getUid()))
					.source(new V1EventSource().component("enhanced-multi-criteria-scheduler"));

			api.createNamespacedEvent(pod.getMetadata().getNamespace(), event, null, null, null, null);

			logger.info(String.format("Successfully bound pod %s to node %s", pod.getMetadata().getName(), nodeName));

		} catch (ApiException e) {
			logger.log(Level.SEVERE, String.format("Failed to bind pod %s to node %s: %s", pod.getMetadata().getName(),
					nodeName, e.getMessage()), e);
			throw e;
		}
	}

	// ============= TASK AND NODE METRICS CLASSES (unchanged) =============

	private static class Task {
		private final String name;
		private final int linesOfCode;
		private final Map<String, Quantity> resourceRequests;
		private final Map<String, String> annotations;

		public Task(V1Pod pod) {
			this.name = pod.getMetadata().getName();
			this.linesOfCode = calculateLinesOfCode(pod);
			this.resourceRequests = getResourceRequests(pod);
			this.annotations = pod.getMetadata().getAnnotations() != null
					? new HashMap<>(pod.getMetadata().getAnnotations())
					: new HashMap<>();
		}

		private static int calculateLinesOfCode(V1Pod pod) {
			if (pod.getMetadata() == null || pod.getMetadata().getAnnotations() == null) {
				return 100; // Default value
			}

			Map<String, String> annotations = pod.getMetadata().getAnnotations();

			if (annotations.containsKey("task.source")) {
				String sourceCode = annotations.get("task.source");
				return countNonEmptyLines(sourceCode);
			}

			if (annotations.containsKey("task.loc")) {
				try {
					return Integer.parseInt(annotations.get("task.loc"));
				} catch (NumberFormatException e) {
					return 100;
				}
			}

			return 100; // Default
		}

		private static int countNonEmptyLines(String sourceCode) {
			if (sourceCode == null || sourceCode.trim().isEmpty()) {
				return 0;
			}
			return (int) Arrays.stream(sourceCode.split("\n")).map(String::trim).filter(line -> !line.isEmpty())
					.filter(line -> !line.startsWith("//")).filter(line -> !line.startsWith("/*"))
					.filter(line -> !line.startsWith("*")).count();
		}

		private static Map<String, Quantity> getResourceRequests(V1Pod pod) {
			if (pod.getSpec() == null || pod.getSpec().getContainers() == null
					|| pod.getSpec().getContainers().isEmpty()
					|| pod.getSpec().getContainers().get(0).getResources() == null
					|| pod.getSpec().getContainers().get(0).getResources().getRequests() == null) {
				return new HashMap<>();
			}
			return pod.getSpec().getContainers().get(0).getResources().getRequests();
		}

		public int getLinesOfCode() {
			return this.linesOfCode;
		}

		public long calculateInstructions() {
			String workloadType = this.annotations.getOrDefault("task.type", "default");
			String workloadSize = this.annotations.getOrDefault("task.size", "small");

			int baseMultiplier = 10;

			switch (workloadType.toLowerCase()) {
			case "compute_intensive":
				baseMultiplier = 15;
				break;
			case "io_intensive":
				baseMultiplier = 8;
				break;
			case "memory_intensive":
				baseMultiplier = 12;
				break;
			}

			double scalingFactor = 1.0;
			switch (workloadSize.toLowerCase()) {
			case "small":
				scalingFactor = 1.0;
				break;
			case "scalable":
				scalingFactor = 1.5;
				break;
			case "distributed":
				scalingFactor = 2.5;
				break;
			}

			return (long) (this.linesOfCode * baseMultiplier * scalingFactor);
		}
	}

	private static class NodeMetrics {
		private final V1Node node;
		private final double mips;
		private final double tdp;
		private final double availableCores;
		private final double availableMemory;
		private final double totalCores;
		private final double totalMemory;
		private static final double IDLE_POWER_RATIO = 0.3;

		public NodeMetrics(V1Node node) throws ApiException {
			this.node = node;
			Map<String, String> labels = node.getMetadata().getLabels();
			Map<String, Quantity> allocatable = node.getStatus().getAllocatable();

			this.totalCores = allocatable.containsKey("cpu") ? parseQuantity(allocatable.get("cpu")) : 1.0;
			this.totalMemory = allocatable.containsKey("memory") ? parseQuantity(allocatable.get("memory")) : 1024.0;

			List<V1Pod> nodePods = getPodsByNodeName(node.getMetadata().getName());
			double usedCpu = nodePods.stream().mapToDouble(pod -> getResourceRequest(pod, "cpu")).sum();
			double usedMemory = nodePods.stream().mapToDouble(pod -> getResourceRequest(pod, "memory")).sum();

			this.availableCores = Math.max(0.0, this.totalCores - usedCpu);
			this.availableMemory = Math.max(0.0, this.totalMemory - usedMemory);

			double efficiency = calculateEfficiencyMultiplier(labels);
			this.mips = this.totalCores * 2000 * efficiency;
			this.tdp = this.totalCores * 25 * efficiency;
		}

		private double calculateEfficiencyMultiplier(Map<String, String> labels) {
			if (labels == null) {
				return 1.0;
			}

			double multiplier = 1.0;
			if (labels.containsKey("feature.node.kubernetes.io/cpu-hardware_multithreading"))
				multiplier *= 1.2;
			if (labels.containsKey("feature.node.kubernetes.io/cpu-cpuid.AVX"))
				multiplier *= 1.1;
			if (labels.containsKey("feature.node.kubernetes.io/cpu-cpuid.AVX2"))
				multiplier *= 1.15;
			if (labels.containsKey("feature.node.kubernetes.io/cpu-cpuid.FMA3"))
				multiplier *= 1.1;

			return multiplier;
		}

		public double estimateExecutionTime(Task task) {
			if (task == null || task.getLinesOfCode() == 0) {
				return 0.0;
			}

			double totalInstructions = task.calculateInstructions();
			Quantity cpuRequest = task.resourceRequests.get("cpu");
			double requestedCores = cpuRequest != null ? parseQuantity(cpuRequest) : 1.0;
			double coreRatio = Math.min(requestedCores / totalCores, 1.0);
			double effectiveMips = mips * coreRatio;

			return effectiveMips > 0 ? totalInstructions / effectiveMips : Double.MAX_VALUE;
		}

		public double estimateEnergy(Task task) {
			if (task == null)
				return 0.0;

			double executionTime = estimateExecutionTime(task);
			Quantity cpuRequest = task.resourceRequests.get("cpu");
			double requestedCores = cpuRequest != null ? parseQuantity(cpuRequest) : 1.0;

			double coreRatio = requestedCores / totalCores;
			double activeEnergy = tdp * coreRatio * executionTime;
			double idleEnergy = tdp * IDLE_POWER_RATIO * (1 - coreRatio) * executionTime;

			return activeEnergy + idleEnergy;
		}
	}
}