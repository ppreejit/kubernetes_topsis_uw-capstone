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
 * A Kubernetes Pod scheduler that implements the TOPSIS (Technique for Order of
 * Preference by Similarity to Ideal Solution) multi-criteria decision-making
 * method to optimize pod placement based on multiple criteria.
 * 
 * This scheduler considers factors such as: - Execution time - Energy
 * consumption - Available CPU cores - Available memory - Resource balance
 * 
 * The scheduler runs as a standalone Java application and interacts with the
 * Kubernetes API to find unscheduled pods and bind them to appropriate nodes.
 */
public class TopsisScheduler {
	private static final Logger logger = Logger.getLogger(TopsisScheduler.class.getName());
	private static CoreV1Api api; // Kubernetes API client for core resources

	/**
	 * Class to track scheduling metrics for each pod. These metrics are used for
	 * performance analysis and are stored as pod annotations.
	 */
	private static class SchedulingMetrics {
		private final String podName;
		private final Duration totalTime; // Total time taken for scheduling
		private final Duration topsisTime; // Time spent in TOPSIS calculation
		private final Duration bindingTime; // Time spent binding the pod to a node
		private final double energyConsumedJoules; // Estimated energy consumption

		/**
		 * Constructor for SchedulingMetrics.
		 * 
		 * @param podName              Pod name
		 * @param totalTime            Total scheduling time
		 * @param topsisTime           Time spent in TOPSIS calculation
		 * @param bindingTime          Time spent binding pod to node
		 * @param energyConsumedJoules Estimated energy consumption in Joules
		 */
		public SchedulingMetrics(String podName, Duration totalTime, Duration topsisTime, Duration bindingTime,
				double energyConsumedJoules) {
			this.podName = podName;
			this.totalTime = totalTime;
			this.topsisTime = topsisTime;
			this.bindingTime = bindingTime;
			this.energyConsumedJoules = energyConsumedJoules;
		}

		/**
		 * Logs all scheduling metrics to the logger.
		 */
		public void logMetrics() {
			double energyKilojoules = energyConsumedJoules / 1000.0;

			logger.info(String.format("Pod %s scheduling metrics:", podName));
			logger.info(String.format("- Total scheduling time: %d ms", totalTime.toMillis()));
			logger.info(String.format("- TOPSIS calculation time: %d ms", topsisTime.toMillis()));
			logger.info(String.format("- Binding time: %d ms", bindingTime.toMillis()));
			logger.info(String.format("- Energy consumption: %.4f kJ", energyKilojoules));
		}
	}

	/**
	 * Main method to run the TOPSIS scheduler. Sets up the Kubernetes client and
	 * runs the scheduling loop.
	 * 
	 * @param args Command line arguments (not used)
	 * @throws IOException If there's an error with I/O operations
	 */
	public static void main(String[] args) throws IOException {
		try {
			// Load logging configuration
			InputStream loggingConfigStream = TopsisScheduler.class.getClassLoader()
					.getResourceAsStream("logging.properties");
			if (loggingConfigStream != null) {
				LogManager.getLogManager().readConfiguration(loggingConfigStream);
				logger.info("Logging configuration loaded successfully");
			} else {
				logger.warning("Could not find logging.properties file");
			}

			logger.info("Initializing TOPSIS Scheduler...");

			// Initialize Kubernetes API client
			ApiClient client = ClientBuilder.cluster().build();
			client.setConnectTimeout(60000); // 60 seconds connection timeout
			client.setReadTimeout(60000); // 60 seconds read timeout
			api = new CoreV1Api(client);

			logger.info("Kubernetes client initialized successfully");

			// Main scheduling loop
			while (true) {
				try {
					logger.info("Starting scheduling cycle");
					// Get all pods that need scheduling
					List<V1Pod> unscheduledPods = getUnscheduledPods(api);

					if (!unscheduledPods.isEmpty()) {
						logger.info(String.format("Found %d unscheduled pods", unscheduledPods.size()));
						// Get all available nodes for scheduling
						List<V1Node> nodes = getAvailableNodes(api);

						if (nodes.isEmpty()) {
							logger.warning("No available nodes found in the cluster");
							continue;
						}

						logger.info(String.format("Found %d available nodes for scheduling", nodes.size()));

						// Process each unscheduled pod
						for (V1Pod pod : unscheduledPods) {
							try {
								// Track scheduling start time
								Instant schedulingStart = Instant.now();
								logger.info(String.format("Starting scheduling process for pod: %s at %s",
										pod.getMetadata().getName(), schedulingStart));

								// Run TOPSIS algorithm to find the best node
								TopsisResult result = topsisSchedule(nodes, pod);
								String bestNode = result.getSelectedNode();
								double energyEstimate = result.getEstimatedEnergy();

								// Track binding start time
								Instant bindingStart = Instant.now();
								Duration topsisTime = Duration.between(schedulingStart, bindingStart);
								logger.info(String.format("TOPSIS calculation completed for pod %s in %d ms",
										pod.getMetadata().getName(), topsisTime.toMillis()));

								bindPodToNode(api, pod, bestNode);

								// Calculate timing metrics
								Instant schedulingEnd = Instant.now();
								Duration totalTime = Duration.between(schedulingStart, schedulingEnd);
								Duration bindingTime = Duration.between(bindingStart, schedulingEnd);

								// Log metrics using our metrics class
								SchedulingMetrics metrics = new SchedulingMetrics(pod.getMetadata().getName(),
										totalTime, topsisTime, bindingTime, energyEstimate);
								metrics.logMetrics();

								// Store metrics in pod annotations for later retrieval
								storeSchedulingMetrics(pod, metrics);

							} catch (Exception e) {
								logger.log(Level.SEVERE, String.format("Failed to schedule pod %s: %s",
										pod.getMetadata().getName(), e.getMessage()), e);
							}
						}
					} else {
						logger.fine("No unscheduled pods found");
					}

					// Wait before next scheduling cycle
					logger.fine("Sleeping for 10 seconds before next scheduling cycle");
					Thread.sleep(10000);

				} catch (InterruptedException e) {
					logger.warning("Scheduler interrupted: " + e.getMessage());
					Thread.currentThread().interrupt();
					break;
				} catch (ApiException e) {
					logger.severe("Kubernetes API error: " + e.getMessage());
					logger.severe("Response Body: " + e.getResponseBody());
					Thread.sleep(30000); // Longer sleep on API error
				} catch (Exception e) {
					logger.severe("Unexpected error: " + e.getMessage());
					Thread.sleep(30000);
				}
			}
		} catch (Exception e) {
			logger.severe("Fatal error in scheduler: " + e.getMessage());
			System.exit(1); // Exit with error code
		}
	}

	/**
	 * Stores scheduling metrics as annotations on the pod. This allows for later
	 * analysis of scheduling performance.
	 * 
	 * @param pod     The pod to annotate
	 * @param metrics The scheduling metrics to store
	 */
	private static void storeSchedulingMetrics(V1Pod pod, SchedulingMetrics metrics) {
		try {
			// Create annotations map
			Map<String, String> annotations = new HashMap<>();
			annotations.put("scheduler.metrics.totalTimeMs", String.valueOf(metrics.totalTime.toMillis()));
			annotations.put("scheduler.metrics.topsisTimeMs", String.valueOf(metrics.topsisTime.toMillis()));
			annotations.put("scheduler.metrics.bindingTimeMs", String.valueOf(metrics.bindingTime.toMillis()));
			annotations.put("scheduler.metrics.energyKJ", String.format("%.4f", metrics.energyConsumedJoules / 1000.0));

			// Create a JSON patch for the pod to add annotations
			V1Patch patch = new V1Patch(String.format("{\"metadata\":{\"annotations\":{"
					+ "\"scheduler.metrics.totalTimeMs\":\"%d\"," + "\"scheduler.metrics.topsisTimeMs\":\"%d\","
					+ "\"scheduler.metrics.bindingTimeMs\":\"%d\"," + "\"scheduler.metrics.energyKJ\":\"%.4f\"" + "}}}",
					metrics.totalTime.toMillis(), metrics.topsisTime.toMillis(), metrics.bindingTime.toMillis(),
					metrics.energyConsumedJoules / 1000.0));

			// Apply the patch to the pod
			api.patchNamespacedPod(pod.getMetadata().getName(), pod.getMetadata().getNamespace(), patch, null, null,
					null, null, null);

			logger.info(String.format("Stored scheduling metrics in pod %s annotations", pod.getMetadata().getName()));
		} catch (ApiException e) {
			logger.warning(String.format("Failed to store metrics in pod annotations: %s", e.getMessage()));
		}
	}

	/**
	 * Retrieves all unscheduled pods that should be scheduled by this scheduler.
	 * Only considers pods with schedulerName set to "topsis-scheduler".
	 * 
	 * @param api Kubernetes API client
	 * @return List of unscheduled pods
	 * @throws ApiException If there's an error calling the Kubernetes API
	 */
	private static List<V1Pod> getUnscheduledPods(CoreV1Api api) throws ApiException {
		Instant start = Instant.now();
		logger.info("Searching for unscheduled pods...");

		// Get all pods across all namespaces
		V1PodList podList = api.listPodForAllNamespaces(null, null, null, null, null, null, null, null, null, null);

		// Filter for unscheduled pods with topsis-scheduler as scheduler name
		List<V1Pod> unscheduledPods = podList.getItems().stream()
				.filter(pod -> pod.getSpec().getNodeName() == null && pod.getSpec().getSchedulerName() != null
						&& pod.getSpec().getSchedulerName().equals("topsis-scheduler"))
				.collect(Collectors.toList());

		Duration duration = Duration.between(start, Instant.now());
		logger.info(String.format("Found %d pods with scheduler name 'topsis-scheduler' in %d ms",
				unscheduledPods.size(), duration.toMillis()));
		return unscheduledPods;
	}

	/**
	 * Retrieves all available nodes in the cluster. Only considers nodes with Ready
	 * condition set to True.
	 * 
	 * @param api Kubernetes API client
	 * @return List of available nodes
	 * @throws ApiException If there's an error calling the Kubernetes API
	 */
	private static List<V1Node> getAvailableNodes(CoreV1Api api) throws ApiException {
		Instant start = Instant.now();
		logger.info("Getting available nodes...");

		// Get all nodes in the cluster
		V1NodeList nodeList = api.listNode(null, null, null, null, null, null, null, null, null, null);

		// Filter for nodes with Ready condition = True
		List<V1Node> availableNodes = nodeList.getItems().stream()
				.filter(node -> node.getStatus().getConditions().stream().anyMatch(
						condition -> condition.getType().equals("Ready") && condition.getStatus().equals("True")))
				.collect(Collectors.toList());

		Duration duration = Duration.between(start, Instant.now());
		logger.info(String.format("Found %d ready nodes in %d ms", availableNodes.size(), duration.toMillis()));
		return availableNodes;
	}

	/**
	 * Parses Kubernetes resource quantities (like CPU and memory) to double values.
	 * Handles various formats and units (m, Ki, Mi, Gi, etc.)
	 * 
	 * @param quantity The resource quantity to parse
	 * @return The parsed value as a double
	 */
	private static double parseQuantity(Quantity quantity) {
		if (quantity == null) {
			return 0.0;
		}

		try {
			// For CPU resources, use the Number value directly
			if (quantity.getFormat().equals("DecimalSI")) {
				return quantity.getNumber().doubleValue();
			}

			String value = quantity.toSuffixedString();
			logger.fine("Parsing quantity: " + value);

			// Handle CPU millicores (e.g., "100m" = 0.1 CPU)
			if (value.endsWith("m")) {
				return Double.parseDouble(value.substring(0, value.length() - 1)) / 1000.0;
			}

			// Extract numeric part and units
			String numericPart = value.replaceAll("[^\\d.]", "");
			double number = Double.parseDouble(numericPart);

			// Handle memory units - always convert to bytes
			// Binary units (Ki, Mi, Gi, Ti)
			if (value.endsWith("Ki")) {
				return number * 1024;
			} else if (value.endsWith("Mi")) {
				return number * 1024 * 1024;
			} else if (value.endsWith("Gi")) {
				return number * 1024 * 1024 * 1024;
			} else if (value.endsWith("Ti")) {
				return number * 1024 * 1024 * 1024 * 1024;
			}
			// Decimal units (K, M, G, T)
			else if (value.endsWith("K")) {
				return number * 1000;
			} else if (value.endsWith("M")) {
				return number * 1000 * 1000;
			} else if (value.endsWith("G")) {
				return number * 1000 * 1000 * 1000;
			} else if (value.endsWith("T")) {
				return number * 1000 * 1000 * 1000 * 1000;
			}

			// If no units, return as is
			return number;
		} catch (Exception e) {
			logger.warning(
					String.format("Failed to parse quantity '%s': %s", quantity.toSuffixedString(), e.getMessage()));
			return 0.0;
		}
	}

	/**
	 * Gets the current CPU usage on a node by summing the CPU requests of all pods
	 * on the node.
	 * 
	 * @param node The node to check
	 * @return The total CPU usage
	 * @throws ApiException If there's an error calling the Kubernetes API
	 */
	private static double getCurrentCpuUsage(V1Node node) throws ApiException {
		// Get all pods running on this node
		String nodeName = node.getMetadata().getName();
		List<V1Pod> nodePods = getPodsByNodeName(nodeName);

		// Sum up CPU requests from all pods
		return nodePods.stream().mapToDouble(pod -> getResourceRequest(pod, "cpu")).sum();
	}

	/**
	 * Gets the current memory usage on a node by summing the memory requests of all
	 * pods on the node.
	 * 
	 * @param node The node to check
	 * @return The total memory usage
	 * @throws ApiException If there's an error calling the Kubernetes API
	 */
	private static double getCurrentMemoryUsage(V1Node node) throws ApiException {
		// Get all pods running on this node
		String nodeName = node.getMetadata().getName();
		List<V1Pod> nodePods = getPodsByNodeName(nodeName);

		// Sum up memory requests from all pods
		return nodePods.stream().mapToDouble(pod -> getResourceRequest(pod, "memory")).sum();
	}

	/**
	 * Filters a list of nodes to only include those with sufficient resources to
	 * run a given pod.
	 * 
	 * @param nodes List of all available nodes
	 * @param pod   The pod to be scheduled
	 * @return Filtered list of eligible nodes
	 * @throws ApiException If there's an error calling the Kubernetes API
	 */
	private static List<V1Node> getEligibleNodes(List<V1Node> nodes, V1Pod pod) throws ApiException {
		List<V1Node> eligibleNodes = new ArrayList<>();
		double requiredCpu = getResourceRequest(pod, "cpu");
		double requiredMemoryBytes = getResourceRequest(pod, "memory");
		double requiredMemoryMB = requiredMemoryBytes / (1024 * 1024);

		logger.info(String.format("Pod %s resource requirements:", pod.getMetadata().getName()));
		logger.info(String.format("- CPU: %.3f cores", requiredCpu));
		logger.info(String.format("- Memory: %.2f MB", requiredMemoryMB));

		// Check each node for eligibility
		for (V1Node node : nodes) {
			try {
				Map<String, Quantity> allocatable = node.getStatus().getAllocatable();
				String nodeName = node.getMetadata().getName();

				// Get node capacity
				double nodeCpu = parseQuantity(allocatable.get("cpu"));
				double nodeMemoryBytes = parseQuantity(allocatable.get("memory"));

				// Get current usage from all pods on the node
				List<V1Pod> nodePods = getPodsByNodeName(nodeName);
				double usedCpu = 0.0;
				double usedMemoryBytes = 0.0;

				// Sum resource usage from all running or pending pods
				for (V1Pod existingPod : nodePods) {
					if (existingPod.getStatus() != null && ("Running".equals(existingPod.getStatus().getPhase())
							|| "Pending".equals(existingPod.getStatus().getPhase()))) {
						usedCpu += getResourceRequest(existingPod, "cpu");
						usedMemoryBytes += getResourceRequest(existingPod, "memory");
					}
				}

				// Calculate available resources
				double availableCpu = nodeCpu - usedCpu;
				double availableMemoryBytes = nodeMemoryBytes - usedMemoryBytes;

				// Log node status
				logger.info(String.format("\nEvaluating node %s:", nodeName));
				logger.info(String.format("- Ready status: %s",
						node.getStatus().getConditions().stream().filter(c -> c.getType().equals("Ready")).findFirst()
								.map(c -> c.getStatus()).orElse("unknown")));
				logger.info(String.format("- CPU: total=%.3f, used=%.3f, available=%.3f cores", nodeCpu, usedCpu,
						availableCpu));
				logger.info(String.format("- Memory: total=%.2f, used=%.2f, available=%.2f MB",
						nodeMemoryBytes / (1024 * 1024), usedMemoryBytes / (1024 * 1024),
						availableMemoryBytes / (1024 * 1024)));

				// Check if node has enough resources
				boolean hasEnoughCpu = availableCpu >= requiredCpu;
				boolean hasEnoughMemory = availableMemoryBytes >= requiredMemoryBytes;

				if (hasEnoughCpu && hasEnoughMemory) {
					logger.info(String.format("Node %s is eligible for pod %s", nodeName, pod.getMetadata().getName()));
					eligibleNodes.add(node);
				} else {
					logger.warning(String.format("Node %s is not eligible: insufficient resources", nodeName));
					logger.warning(String.format(
							"Required: CPU=%.3f (available %.3f), Memory=%.2f MB (available %.2f MB)", requiredCpu,
							availableCpu, requiredMemoryBytes / (1024 * 1024), availableMemoryBytes / (1024 * 1024)));
				}
			} catch (ApiException e) {
				logger.warning(String.format("Failed to check resources for node %s: %s", node.getMetadata().getName(),
						e.getMessage()));
			}
		}

		if (eligibleNodes.isEmpty()) {
			logger.warning("No eligible nodes found for pod " + pod.getMetadata().getName());
		} else {
			logger.info(String.format("Found %d eligible nodes for pod %s", eligibleNodes.size(),
					pod.getMetadata().getName()));
		}

		return eligibleNodes;
	}

	/**
	 * Class to hold results from TOPSIS calculation.
	 */
	private static class TopsisResult {
		private final String selectedNode; // Name of the selected node
		private final double relativeCloseness; // TOPSIS relative closeness score
		private final double estimatedEnergy; // Estimated energy consumption

		public TopsisResult(String selectedNode, double relativeCloseness, double estimatedEnergy) {
			this.selectedNode = selectedNode;
			this.relativeCloseness = relativeCloseness;
			this.estimatedEnergy = estimatedEnergy;
		}

		public String getSelectedNode() {
			return selectedNode;
		}

		public double getRelativeCloseness() {
			return relativeCloseness;
		}

		public double getEstimatedEnergy() {
			return estimatedEnergy;
		}
	}

	/**
	 * Implements the TOPSIS algorithm to find the best node for a pod. TOPSIS
	 * (Technique for Order of Preference by Similarity to Ideal Solution) is a
	 * multi-criteria decision-making method.
	 * 
	 * @param nodes List of all available nodes
	 * @param pod   The pod to be scheduled
	 * @return TopsisResult containing the selected node and metrics
	 * @throws ApiException If there's an error calling the Kubernetes API
	 */
	private static TopsisResult topsisSchedule(List<V1Node> nodes, V1Pod pod) throws ApiException {
		Instant start = Instant.now();

		if (nodes.isEmpty()) {
			throw new IllegalArgumentException("No nodes available for scheduling");
		}

		// Get eligible nodes first (nodes with sufficient resources)
		List<V1Node> eligibleNodes = getEligibleNodes(nodes, pod);

		if (eligibleNodes.isEmpty()) {
			throw new ApiException("No nodes have sufficient resources for pod " + pod.getMetadata().getName()
					+ String.format(" (Required: CPU=%.3f, Memory=%d)", getResourceRequest(pod, "cpu"),
							(long) getResourceRequest(pod, "memory")));
		}

		// Create a task representation of the pod
		Task task = new Task(pod);
		logger.info(
				String.format("Starting TOPSIS calculation for task %s on %d nodes", task.name, eligibleNodes.size()));

		// TOPSIS criteria definitions
		// Criteria: ExecutionTime, Energy, CoreAvail, MemAvail, ResourceBalance
		int numCriteria = 5;
		// Weights for each criterion (must sum to 1.0)
		double[] weights = { 0.2, 0.2, 0.2, 0.2, 0.2 };
		// Decision matrix: rows=nodes, columns=criteria
		double[][] decisionMatrix = new double[eligibleNodes.size()][numCriteria];

		// Track estimated energy for the best node
		double[] nodeEnergyEstimates = new double[eligibleNodes.size()];

		// Step 1: Create decision matrix using eligible nodes
		Instant matrixStart = Instant.now();
		for (int i = 0; i < eligibleNodes.size(); i++) {
			try {
				NodeMetrics metrics = new NodeMetrics(eligibleNodes.get(i));

				// Calculate metrics for each node
				double executionTime = metrics.estimateExecutionTime(task);
				double energy = metrics.estimateEnergy(task);
				nodeEnergyEstimates[i] = energy; // Store the energy estimate
				double resourceBalance = Math.abs(
						metrics.availableCores / metrics.totalCores - metrics.availableMemory / metrics.totalMemory);

				// Populate decision matrix (negative for minimization criteria)
				decisionMatrix[i][0] = -executionTime; // Minimize execution time
				decisionMatrix[i][1] = -energy; // Minimize energy
				decisionMatrix[i][2] = metrics.availableCores; // Maximize core availability
				decisionMatrix[i][3] = metrics.availableMemory;// Maximize memory availability
				decisionMatrix[i][4] = -resourceBalance; // Minimize resource imbalance

				logger.info(String.format(
						"Node %s metrics - Execution Time: %.2fs, Energy: %.2fJ, "
								+ "Available Cores: %.2f, Available Memory: %.2f, Balance: %.2f",
						eligibleNodes.get(i).getMetadata().getName(), executionTime, energy, metrics.availableCores,
						metrics.availableMemory, resourceBalance));
			} catch (Exception e) {
				logger.warning(String.format("Error calculating metrics for node %s: %s",
						eligibleNodes.get(i).getMetadata().getName(), e.getMessage()));
				Arrays.fill(decisionMatrix[i], 0.0);
			}
		}

		Duration matrixTime = Duration.between(matrixStart, Instant.now());

		// Step 2: Normalize the decision matrix
		Instant normStart = Instant.now();
		double[] columnSums = new double[numCriteria];
		// Calculate Euclidean norm for each column
		for (int j = 0; j < numCriteria; j++) {
			for (int i = 0; i < eligibleNodes.size(); i++) {
				columnSums[j] += Math.pow(decisionMatrix[i][j], 2);
			}
			columnSums[j] = Math.sqrt(columnSums[j]);

			// Avoid division by zero
			if (columnSums[j] == 0) {
				columnSums[j] = 1;
				logger.warning(String.format("Column %d has sum of zero, using 1 to avoid division by zero", j));
			}
		}

		// Apply weights during normalization
		for (int i = 0; i < eligibleNodes.size(); i++) {
			for (int j = 0; j < numCriteria; j++) {
				decisionMatrix[i][j] = (decisionMatrix[i][j] / columnSums[j]) * weights[j];
			}
		}
		Duration normTime = Duration.between(normStart, Instant.now());
		// Step 3: Calculate ideal and negative-ideal solutions
		Instant idealStart = Instant.now();
		double[] idealSolution = new double[numCriteria];
		double[] negativeIdealSolution = new double[numCriteria];
		Arrays.fill(idealSolution, Double.MIN_VALUE);
		Arrays.fill(negativeIdealSolution, Double.MAX_VALUE);

		// Find the maximum and minimum values for each criterion
		for (int j = 0; j < numCriteria; j++) {
			for (int i = 0; i < eligibleNodes.size(); i++) {
				idealSolution[j] = Math.max(idealSolution[j], decisionMatrix[i][j]);
				negativeIdealSolution[j] = Math.min(negativeIdealSolution[j], decisionMatrix[i][j]);
			}
		}
		Duration idealTime = Duration.between(idealStart, Instant.now());

		// Step 4: Calculate separations (distances) from ideal and negative-ideal
		// solutions
		Instant sepStart = Instant.now();
		double[] separationIdeal = new double[eligibleNodes.size()]; // Distance from ideal
		double[] separationNegativeIdeal = new double[eligibleNodes.size()]; // Distance from negative ideal

		// Calculate Euclidean distances
		for (int i = 0; i < eligibleNodes.size(); i++) {
			for (int j = 0; j < numCriteria; j++) {
				separationIdeal[i] += Math.pow(decisionMatrix[i][j] - idealSolution[j], 2);
				separationNegativeIdeal[i] += Math.pow(decisionMatrix[i][j] - negativeIdealSolution[j], 2);
			}
			separationIdeal[i] = Math.sqrt(separationIdeal[i]);
			separationNegativeIdeal[i] = Math.sqrt(separationNegativeIdeal[i]);
		}
		Duration sepTime = Duration.between(sepStart, Instant.now());

		// Step 5: Calculate relative closeness and select best node
		Instant selectionStart = Instant.now();
		double maxCloseness = Double.MIN_VALUE;
		int bestNodeIndex = 0;

		// Find node with highest relative closeness (closest to ideal solution)
		for (int i = 0; i < eligibleNodes.size(); i++) {
			double separationSum = separationIdeal[i] + separationNegativeIdeal[i];
			double relativeCloseness = separationSum == 0 ? 0 : separationNegativeIdeal[i] / separationSum;

			logger.fine(String.format("Node %s relative closeness: %f", eligibleNodes.get(i).getMetadata().getName(),
					relativeCloseness));

			if (relativeCloseness > maxCloseness) {
				maxCloseness = relativeCloseness;
				bestNodeIndex = i;
			}
		}

		// Get final selected node and its energy estimate
		String selectedNode = eligibleNodes.get(bestNodeIndex).getMetadata().getName();
		double estimatedEnergy = nodeEnergyEstimates[bestNodeIndex];
		Duration selectionTime = Duration.between(selectionStart, Instant.now());
		Duration totalTime = Duration.between(start, Instant.now());

		// Log timing breakdown for performance analysis
		logger.info(
				String.format("TOPSIS calculation completed in %d ms with timing breakdown:", totalTime.toMillis()));
		logger.info(String.format("- Matrix creation: %d ms", matrixTime.toMillis()));
		logger.info(String.format("- Normalization: %d ms", normTime.toMillis()));
		logger.info(String.format("- Ideal solutions: %d ms", idealTime.toMillis()));
		logger.info(String.format("- Separations: %d ms", sepTime.toMillis()));
		logger.info(String.format("- Final selection: %d ms", selectionTime.toMillis()));
		logger.info(String.format("Selected node %s with relative closeness %f", selectedNode, maxCloseness));
		logger.info(String.format("Estimated energy consumption: %.4f J (%.4f kJ)", estimatedEnergy,
				estimatedEnergy / 1000.0));

		return new TopsisResult(selectedNode, maxCloseness, estimatedEnergy);
	}

	/**
	 * Gets the total requested resources for a specific resource type from a pod.
	 * Sums up requests from all containers in the pod.
	 * 
	 * @param pod          The pod to check
	 * @param resourceName The resource name (e.g., "cpu", "memory")
	 * @return The total requested resource amount
	 */
	private static double getResourceRequest(V1Pod pod, String resourceName) {
		if (pod.getSpec() == null || pod.getSpec().getContainers() == null) {
			return 0.0;
		}

		// Sum resource requests from all containers
		return pod.getSpec().getContainers().stream()
				.filter(container -> container.getResources() != null && container.getResources().getRequests() != null
						&& container.getResources().getRequests().containsKey(resourceName))
				.mapToDouble(container -> {
					Quantity quantity = container.getResources().getRequests().get(resourceName);
					return parseQuantity(quantity);
				}).sum();
	}

	/**
	 * Gets all pods scheduled to a specific node.
	 * 
	 * @param nodeName The name of the node
	 * @return List of pods on the node
	 * @throws ApiException If there's an error calling the Kubernetes API
	 */
	private static List<V1Pod> getPodsByNodeName(String nodeName) throws ApiException {
		String fieldSelector = String.format("spec.nodeName=%s", nodeName);
		V1PodList podList = api.listPodForAllNamespaces(null, null, fieldSelector, null, null, null, null, null, null,
				null);
		return podList.getItems();
	}

	/**
	 * Binds a pod to a node by creating a binding object and posting it to the
	 * Kubernetes API. Also creates an event to record the scheduling decision.
	 * 
	 * @param api      Kubernetes API client
	 * @param pod      The pod to bind
	 * @param nodeName The name of the node to bind to
	 * @throws ApiException If there's an error calling the Kubernetes API
	 */
	private static void bindPodToNode(CoreV1Api api, V1Pod pod, String nodeName) throws ApiException {
		Instant bindStart = Instant.now();
		logger.info(String.format("Starting pod binding process at %s", bindStart));

		try {
			// Create binding object
			V1Binding binding = new V1Binding().metadata(new V1ObjectMeta().name(pod.getMetadata().getName()))
					.target(new V1ObjectReference().apiVersion("v1").kind("Node").name(nodeName));

			// API call to create the pod binding
			api.createNamespacedPodBinding(pod.getMetadata().getName(), pod.getMetadata().getNamespace(), binding, null,
					null, null, null);

			// Create event with proper metadata to record the scheduling decision
			V1ObjectMeta eventMetadata = new V1ObjectMeta().name("topsis-scheduled-" + pod.getMetadata().getName() + "-"
					+ UUID.randomUUID().toString().substring(0, 8)).namespace(pod.getMetadata().getNamespace());

			CoreV1Event event = new CoreV1Event().metadata(eventMetadata).type("Normal").reason("Scheduled")
					.message("Pod scheduled by topsis-scheduler")
					.involvedObject(new V1ObjectReference().kind("Pod").name(pod.getMetadata().getName())
							.namespace(pod.getMetadata().getNamespace()).uid(pod.getMetadata().getUid()))
					.source(new V1EventSource().component("topsis-scheduler"));

			api.createNamespacedEvent(pod.getMetadata().getNamespace(), event, null, null, null, null);

			Duration bindTime = Duration.between(bindStart, Instant.now());
			logger.info(String.format("Successfully bound pod %s to node %s in %d ms", pod.getMetadata().getName(),
					nodeName, bindTime.toMillis()));

		} catch (ApiException e) {
			Duration errorTime = Duration.between(bindStart, Instant.now());
			logger.log(Level.SEVERE, String.format("Failed to bind pod %s to node %s after %d ms due to API error: %s",
					pod.getMetadata().getName(), nodeName, errorTime.toMillis(), e.getMessage()), e);
			logger.log(Level.SEVERE, "HTTP response body: " + e.getResponseBody());
			logger.log(Level.SEVERE, "HTTP response code: " + e.getCode());
			throw e;
		} catch (Exception e) {
			Duration errorTime = Duration.between(bindStart, Instant.now());
			logger.log(Level.SEVERE,
					String.format("Unexpected error occurred while binding pod %s to node %s after %d ms",
							pod.getMetadata().getName(), nodeName, errorTime.toMillis()),
					e);
			throw new ApiException(e);
		}
	}

	/**
	 * Represents a task to be scheduled, containing information from a Kubernetes
	 * pod. Contains methods for calculating resource requirements and computational
	 * complexity.
	 */
	private static class Task {
		private final String name; // Task name (from pod name)
		private final int linesOfCode; // Code complexity measure
		private final Map<String, Quantity> resourceRequests; // Resource requests from pod
		private final Map<String, String> annotations; // Pod annotations

		/**
		 * Creates a Task from a Kubernetes pod.
		 * 
		 * @param pod The pod to create a task from
		 */
		public Task(V1Pod pod) {
			this.name = pod.getMetadata().getName();
			this.linesOfCode = calculateLinesOfCode(pod);
			this.resourceRequests = getResourceRequests(pod);
			this.annotations = pod.getMetadata().getAnnotations() != null
					? new HashMap<>(pod.getMetadata().getAnnotations())
					: new HashMap<>();
		}

		/**
		 * Calculates lines of code from pod annotations. Can use various sources:
		 * direct code, LOC count, or GitHub repo.
		 * 
		 * @param pod The pod to analyze
		 * @return Lines of code count
		 */
		private static int calculateLinesOfCode(V1Pod pod) {
			if (pod.getMetadata() == null || pod.getMetadata().getAnnotations() == null) {
				logger.warning("No metadata or annotations found for pod " + pod.getMetadata().getName());
				return 0;
			}

			Map<String, String> annotations = pod.getMetadata().getAnnotations();

			// Try to get source code directly from annotations
			if (annotations.containsKey("task.source")) {
				String sourceCode = annotations.get("task.source");
				return countNonEmptyLines(sourceCode);
			}

			// Try to get LOC count from annotations
			if (annotations.containsKey("task.loc")) {
				try {
					return Integer.parseInt(annotations.get("task.loc"));
				} catch (NumberFormatException e) {
					logger.warning("Invalid LOC count in annotation: " + annotations.get("task.loc"));
					return 0;
				}
			}

			// Try to get from GitHub repo
			if (annotations.containsKey("task.github")) {
				String repoInfo = annotations.get("task.github");
				return getLocFromGitHub(repoInfo);
			}

			logger.warning("No LOC information found for pod " + pod.getMetadata().getName());
			return 0;
		}

		/**
		 * Counts non-empty, non-comment lines in source code.
		 * 
		 * @param sourceCode The source code as a string
		 * @return Count of significant lines
		 */
		private static int countNonEmptyLines(String sourceCode) {
			if (sourceCode == null || sourceCode.trim().isEmpty()) {
				return 0;
			}
			// Count lines that are not empty and not comments
			return (int) Arrays.stream(sourceCode.split("\n")).map(String::trim).filter(line -> !line.isEmpty())
					.filter(line -> !line.startsWith("//")).filter(line -> !line.startsWith("/*"))
					.filter(line -> !line.startsWith("*")).count();
		}

		/**
		 * Gets lines of code from a GitHub repository. Currently not implemented.
		 * 
		 * @param repoInfo GitHub repository information
		 * @return Lines of code count (currently always 0)
		 */
		private static int getLocFromGitHub(String repoInfo) {
			logger.warning("GitHub LOC calculation not implemented yet");
			return 0;
		}

		/**
		 * Gets resource requests from a pod.
		 * 
		 * @param pod The pod to analyze
		 * @return Map of resource requests
		 */
		private static Map<String, Quantity> getResourceRequests(V1Pod pod) {
			if (pod.getSpec() == null || pod.getSpec().getContainers() == null
					|| pod.getSpec().getContainers().isEmpty()
					|| pod.getSpec().getContainers().get(0).getResources() == null
					|| pod.getSpec().getContainers().get(0).getResources().getRequests() == null) {
				return new HashMap<>();
			}
			return pod.getSpec().getContainers().get(0).getResources().getRequests();
		}

		/**
		 * Gets the lines of code for this task.
		 * 
		 * @return Lines of code count
		 */
		public int getLinesOfCode() {
			return this.linesOfCode;
		}

		/**
		 * Calculates the estimated number of instructions for this task. Takes into
		 * account workload type and size from annotations.
		 * 
		 * @return Estimated instruction count
		 */
		public long calculateInstructions() {
			String workloadType = this.annotations.getOrDefault("task.type", "default");
			String workloadSize = this.annotations.getOrDefault("task.size", "small");

			int baseMultiplier = 10;

			// Adjust for workload type
			switch (workloadType.toLowerCase()) {
			case "compute_intensive":
				baseMultiplier = 15; // More instructions per line for compute-intensive tasks
				break;
			case "io_intensive":
				baseMultiplier = 8; // Fewer instructions for I/O-bound tasks
				break;
			case "memory_intensive":
				baseMultiplier = 12; // Medium instructions for memory-intensive tasks
				break;
			}

			// Apply scaling factor based on workload size
			double scalingFactor = 1.0;
			switch (workloadSize.toLowerCase()) {
			case "small":
				scalingFactor = 1.0; // Base case
				break;
			case "scalable":
				scalingFactor = 1.5; // 50% more instructions per line for scalable
				break;
			case "distributed":
				scalingFactor = 2.5; // 150% more instructions per line for distributed
				break;
			}

			return (long) (this.linesOfCode * baseMultiplier * scalingFactor);
		}
	}

	/**
	 * Represents the metrics and capabilities of a Kubernetes node. Used for
	 * estimating task execution time and energy consumption.
	 */
	private static class NodeMetrics {
		private final V1Node node; // The Kubernetes node
		private final double mips; // Million Instructions Per Second (performance)
		private final double tdp; // Thermal Design Power (watts)
		private final double availableCores; // Available CPU cores
		private final double availableMemory; // Available memory
		private final double totalCores; // Total CPU cores
		private final double totalMemory; // Total memory
		// Default instructions per line of code if not specified
		private static final double INSTRUCTIONS_PER_LOC = Double
				.parseDouble(System.getProperty("INSTRUCTIONS_PER_LOC", "10"));
		// Ratio of power consumption at idle vs. full load
		private static final double IDLE_POWER_RATIO = 0.3;

		/**
		 * Creates NodeMetrics for a Kubernetes node. Calculates performance metrics and
		 * available resources.
		 * 
		 * @param node The Kubernetes node
		 * @throws ApiException If there's an error calling the Kubernetes API
		 */
		public NodeMetrics(V1Node node) throws ApiException {
			this.node = node;
			Map<String, String> labels = node.getMetadata().getLabels();
			Map<String, Quantity> allocatable = node.getStatus().getAllocatable();

			// Get total resources
			this.totalCores = allocatable.containsKey("cpu") ? parseQuantity(allocatable.get("cpu")) : 1.0;
			this.totalMemory = allocatable.containsKey("memory") ? parseQuantity(allocatable.get("memory")) : 1024.0;

			// Calculate used resources
			List<V1Pod> nodePods = getPodsByNodeName(node.getMetadata().getName());
			double usedCpu = nodePods.stream().mapToDouble(pod -> getResourceRequest(pod, "cpu")).sum();
			double usedMemory = nodePods.stream().mapToDouble(pod -> getResourceRequest(pod, "memory")).sum();

			// Calculate available resources
			this.availableCores = Math.max(0.0, this.totalCores - usedCpu);
			this.availableMemory = Math.max(0.0, this.totalMemory - usedMemory);

			// Calculate performance metrics based on CPU features
			double efficiency = calculateEfficiencyMultiplier(labels);
			this.mips = this.totalCores * 2000 * efficiency; // 2000 MIPS per core baseline
			this.tdp = this.totalCores * 25 * efficiency; // 25 watts per core baseline
		}

		/**
		 * Calculates an efficiency multiplier based on CPU features in node labels.
		 * Rewards nodes with advanced features like hyper-threading, AVX instructions,
		 * etc.
		 * 
		 * @param labels Node labels containing CPU feature information
		 * @return Efficiency multiplier (1.0 = baseline)
		 */
		private double calculateEfficiencyMultiplier(Map<String, String> labels) {
			if (labels == null) {
				return 1.0;
			}

			double multiplier = 1.0;

			// Apply bonuses for CPU features
			if (labels.containsKey("feature.node.kubernetes.io/cpu-hardware_multithreading"))
				multiplier *= 1.2; // 20% bonus for hyper-threading
			if (labels.containsKey("feature.node.kubernetes.io/cpu-cpuid.AVX"))
				multiplier *= 1.1; // 10% bonus for AVX
			if (labels.containsKey("feature.node.kubernetes.io/cpu-cpuid.AVX2"))
				multiplier *= 1.15; // 15% bonus for AVX2
			if (labels.containsKey("feature.node.kubernetes.io/cpu-cpuid.FMA3"))
				multiplier *= 1.1; // 10% bonus for FMA3

			return multiplier;
		}

		/**
		 * Estimates execution time for a task on this node. Based on task complexity
		 * and node performance.
		 * 
		 * @param task The task to execute
		 * @return Estimated execution time in seconds
		 */
		public double estimateExecutionTime(Task task) {
			if (task == null || task.getLinesOfCode() == 0) {
				return 0.0;
			}

			// Calculate total instructions based on task complexity
			double totalInstructions = task.calculateInstructions();
			// Determine how much of the node's compute power will be used
			Quantity cpuRequest = task.resourceRequests.get("cpu");
			double requestedCores = cpuRequest != null ? parseQuantity(cpuRequest) : 1.0;
			double coreRatio = Math.min(requestedCores / totalCores, 1.0);
			// Calculate effective MIPS (accounting for partial core usage)
			double effectiveMips = mips * coreRatio;

			// Time = Instructions / MIPS (but avoid division by zero)
			return effectiveMips > 0 ? totalInstructions / effectiveMips : Double.MAX_VALUE;
		}

		/**
		 * Estimates energy consumption for executing a task on this node. Based on
		 * execution time, TDP, and core utilization.
		 * 
		 * @param task The task to execute
		 * @return Estimated energy consumption in Joules
		 */
		public double estimateEnergy(Task task) {
			if (task == null)
				return 0.0;

			// Calculate execution time and core utilization
			double executionTime = estimateExecutionTime(task);
			Quantity cpuRequest = task.resourceRequests.get("cpu");
			double requestedCores = cpuRequest != null ? parseQuantity(cpuRequest) : 1.0;

			// Calculate energy consumption
			double coreRatio = requestedCores / totalCores;
			// Energy for active cores
			double activeEnergy = tdp * coreRatio * executionTime;
			// Energy for idle cores (uses less power)
			double idleEnergy = tdp * IDLE_POWER_RATIO * (1 - coreRatio) * executionTime;

			return activeEnergy + idleEnergy;
		}
	}
}