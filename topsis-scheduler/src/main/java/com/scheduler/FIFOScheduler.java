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
import java.util.logging.Logger;
import java.util.logging.LogManager;
import java.util.stream.Collectors;
import java.time.Instant;
import java.time.Duration;

/**
 * FIFO (First-In-First-Out) Scheduler
 * 
 * Simple scheduling algorithm that: 1. Processes pods in the order they were
 * created (FIFO queue) 2. Assigns each pod to the first node with sufficient
 * resources 3. No optimization - purely first-come-first-served
 * 
 * Used as baseline comparison for enhanced multi-criteria scheduler.
 */
public class FIFOScheduler {
	private static final Logger logger = Logger.getLogger(FIFOScheduler.class.getName());
	private static CoreV1Api api;

	private static class SchedulingMetrics {
		private final String podName;
		private final Duration totalTime;
		private final Duration selectionTime;
		private final Duration bindingTime;
		private final String selectedNode;

		public SchedulingMetrics(String podName, Duration totalTime, Duration selectionTime, Duration bindingTime,
				String selectedNode) {
			this.podName = podName;
			this.totalTime = totalTime;
			this.selectionTime = selectionTime;
			this.bindingTime = bindingTime;
			this.selectedNode = selectedNode;
		}

		public void logMetrics() {
			logger.info(String.format("Pod %s scheduling metrics:", podName));
			logger.info(String.format("- Algorithm: FIFO"));
			logger.info(String.format("- Selected node: %s", selectedNode));
			logger.info(String.format("- Total scheduling time: %d ms", totalTime.toMillis()));
			logger.info(String.format("- Node selection time: %d ms", selectionTime.toMillis()));
			logger.info(String.format("- Binding time: %d ms", bindingTime.toMillis()));
		}
	}

	public static void main(String[] args) throws IOException {
		try {
			// Load logging configuration
			InputStream loggingConfigStream = FIFOScheduler.class.getClassLoader()
					.getResourceAsStream("logging.properties");
			if (loggingConfigStream != null) {
				LogManager.getLogManager().readConfiguration(loggingConfigStream);
				logger.info("Logging configuration loaded successfully");
			}

			logger.info("Initializing FIFO Scheduler...");

			// Initialize Kubernetes API client
			ApiClient client = ClientBuilder.cluster().build();
			client.setConnectTimeout(60000);
			client.setReadTimeout(60000);
			api = new CoreV1Api(client);

			logger.info("Kubernetes client initialized successfully");

			// Main scheduling loop
			while (true) {
				try {
					logger.fine("Starting scheduling cycle");
					List<V1Pod> unscheduledPods = getUnscheduledPods(api);

					if (!unscheduledPods.isEmpty()) {
						logger.info(String.format("Found %d unscheduled pods", unscheduledPods.size()));

						// Sort by creation timestamp (FIFO)
						unscheduledPods.sort(Comparator.comparing(pod -> pod.getMetadata().getCreationTimestamp()));

						List<V1Node> nodes = getAvailableNodes(api);

						if (nodes.isEmpty()) {
							logger.warning("No available nodes found in the cluster");
							continue;
						}

						logger.info(String.format("Found %d available nodes for scheduling", nodes.size()));

						// Process each pod in FIFO order
						for (V1Pod pod : unscheduledPods) {
							try {
								Instant schedulingStart = Instant.now();

								logger.info(String.format("Processing pod: %s (created at %s)",
										pod.getMetadata().getName(), pod.getMetadata().getCreationTimestamp()));

								// FIFO: Select first node with sufficient resources
								String selectedNode = selectNodeFIFO(nodes, pod);

								Instant bindingStart = Instant.now();
								Duration selectionTime = Duration.between(schedulingStart, bindingStart);

								if (selectedNode == null) {
									logger.warning(String.format("No suitable node found for pod %s",
											pod.getMetadata().getName()));
									continue;
								}

								bindPodToNode(api, pod, selectedNode);

								Instant schedulingEnd = Instant.now();
								Duration totalTime = Duration.between(schedulingStart, schedulingEnd);
								Duration bindingTime = Duration.between(bindingStart, schedulingEnd);

								// Log metrics
								SchedulingMetrics metrics = new SchedulingMetrics(pod.getMetadata().getName(),
										totalTime, selectionTime, bindingTime, selectedNode);
								metrics.logMetrics();

								// Store metrics in pod annotations
								storeSchedulingMetrics(pod, metrics);

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
	 * FIFO node selection: return first node with sufficient resources
	 */
	private static String selectNodeFIFO(List<V1Node> nodes, V1Pod pod) throws ApiException {
		double requiredCpu = getResourceRequest(pod, "cpu");
		double requiredMemory = getResourceRequest(pod, "memory");

		logger.info(String.format("Pod %s requires: CPU=%.3f, Memory=%.2f MB", pod.getMetadata().getName(), requiredCpu,
				requiredMemory / (1024 * 1024)));

		// Iterate through nodes in order - select FIRST with sufficient resources
		for (V1Node node : nodes) {
			try {
				Map<String, Quantity> allocatable = node.getStatus().getAllocatable();
				double nodeCpu = parseQuantity(allocatable.get("cpu"));
				double nodeMemory = parseQuantity(allocatable.get("memory"));

				// Calculate used resources
				List<V1Pod> nodePods = getPodsByNodeName(node.getMetadata().getName());
				double usedCpu = nodePods.stream().filter(p -> p.getStatus() != null
						&& ("Running".equals(p.getStatus().getPhase()) || "Pending".equals(p.getStatus().getPhase())))
						.mapToDouble(p -> getResourceRequest(p, "cpu")).sum();

				double usedMemory = nodePods.stream().filter(p -> p.getStatus() != null
						&& ("Running".equals(p.getStatus().getPhase()) || "Pending".equals(p.getStatus().getPhase())))
						.mapToDouble(p -> getResourceRequest(p, "memory")).sum();

				double availableCpu = nodeCpu - usedCpu;
				double availableMemory = nodeMemory - usedMemory;

				logger.fine(String.format("Node %s: Available CPU=%.3f, Available Memory=%.2f MB",
						node.getMetadata().getName(), availableCpu, availableMemory / (1024 * 1024)));

				// FIFO: Return FIRST node that fits
				if (availableCpu >= requiredCpu && availableMemory >= requiredMemory) {
					logger.info(String.format("Selected node %s (first fit)", node.getMetadata().getName()));
					return node.getMetadata().getName();
				}

			} catch (Exception e) {
				logger.warning(
						String.format("Error checking node %s: %s", node.getMetadata().getName(), e.getMessage()));
			}
		}

		logger.warning(
				String.format("No node found with sufficient resources for pod %s", pod.getMetadata().getName()));
		return null;
	}

	/**
	 * Store scheduling metrics as annotations
	 */
	private static void storeSchedulingMetrics(V1Pod pod, SchedulingMetrics metrics) {
		try {
			V1Patch patch = new V1Patch(String.format(
					"{\"metadata\":{\"annotations\":{" + "\"scheduler.algorithm\":\"FIFO\","
							+ "\"scheduler.node\":\"%s\"," + "\"scheduler.metrics.totalTimeMs\":\"%d\","
							+ "\"scheduler.metrics.selectionTimeMs\":\"%d\","
							+ "\"scheduler.metrics.bindingTimeMs\":\"%d\"" + "}}}",
					metrics.selectedNode, metrics.totalTime.toMillis(), metrics.selectionTime.toMillis(),
					metrics.bindingTime.toMillis()));

			api.patchNamespacedPod(pod.getMetadata().getName(), pod.getMetadata().getNamespace(), patch, null, null,
					null, null, null);

			logger.fine(String.format("Stored metrics for pod %s", pod.getMetadata().getName()));
		} catch (ApiException e) {
			logger.warning(String.format("Failed to store metrics: %s", e.getMessage()));
		}
	}

	// ============= UTILITY METHODS =============

	private static List<V1Pod> getUnscheduledPods(CoreV1Api api) throws ApiException {
		V1PodList podList = api.listPodForAllNamespaces(null, null, null, null, null, null, null, null, null, null,
				null);

		return podList.getItems().stream()
				.filter(pod -> pod.getSpec().getNodeName() == null && pod.getSpec().getSchedulerName() != null
						&& pod.getSpec().getSchedulerName().equals("fifo-scheduler"))
				.collect(Collectors.toList());
	}

	private static List<V1Node> getAvailableNodes(CoreV1Api api) throws ApiException {
		V1NodeList nodeList = api.listNode(null, null, null, null, null, null, null, null, null, null, null);

		return nodeList.getItems().stream()
				.filter(node -> node.getStatus().getConditions().stream().anyMatch(
						condition -> condition.getType().equals("Ready") && condition.getStatus().equals("True")))
				.collect(Collectors.toList());
	}

	private static List<V1Pod> getPodsByNodeName(String nodeName) throws ApiException {
		String fieldSelector = String.format("spec.nodeName=%s", nodeName);
		V1PodList podList = api.listPodForAllNamespaces(null, null, fieldSelector, null, null, null, null, null, null,
				null, null);
		return podList.getItems();
	}

	private static double parseQuantity(Quantity quantity) {
		if (quantity == null)
			return 0.0;

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

			if (value.endsWith("Ki"))
				return number * 1024;
			else if (value.endsWith("Mi"))
				return number * 1024 * 1024;
			else if (value.endsWith("Gi"))
				return number * 1024 * 1024 * 1024;
			else if (value.endsWith("Ti"))
				return number * 1024L * 1024 * 1024 * 1024;

			return number;
		} catch (Exception e) {
			logger.warning(
					String.format("Failed to parse quantity '%s': %s", quantity.toSuffixedString(), e.getMessage()));
			return 0.0;
		}
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

			// Create event
			CoreV1Event event = new CoreV1Event()
					.metadata(new V1ObjectMeta()
							.name("fifo-scheduled-" + pod.getMetadata().getName() + "-"
									+ UUID.randomUUID().toString().substring(0, 8))
							.namespace(pod.getMetadata().getNamespace()))
					.type("Normal").reason("Scheduled").message("Pod scheduled by FIFO scheduler")
					.involvedObject(new V1ObjectReference().kind("Pod").name(pod.getMetadata().getName())
							.namespace(pod.getMetadata().getNamespace()).uid(pod.getMetadata().getUid()))
					.source(new V1EventSource().component("fifo-scheduler"));

			api.createNamespacedEvent(pod.getMetadata().getNamespace(), event, null, null, null, null);

			logger.info(String.format("Successfully bound pod %s to node %s", pod.getMetadata().getName(), nodeName));

		} catch (ApiException e) {
			logger.log(Level.SEVERE, String.format("Failed to bind pod %s to node %s: %s", pod.getMetadata().getName(),
					nodeName, e.getMessage()), e);
			throw e;
		}
	}
}