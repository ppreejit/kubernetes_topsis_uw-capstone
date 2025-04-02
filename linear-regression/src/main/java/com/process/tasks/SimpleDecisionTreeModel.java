package com.process.tasks;

import java.io.*;
import java.util.*;

/**
 * Simple Decision Tree implementation for small datasets - Limited depth
 * (max_depth=3) - Basic binary splitting - Handles categorical and numerical
 * features - Single-threaded processing
 */
public class SimpleDecisionTreeModel {
	private Node root;
	private final int maxDepth;
	private double accuracy;

	// Decision tree node
	private static class Node {
		int featureIndex; // Index of feature to split on
		double threshold; // Threshold for numerical feature
		String category; // Category for categorical feature
		boolean isLeaf; // Is this a leaf node
		double prediction; // Prediction value for leaf node
		Node left; // Left child (true branch)
		Node right; // Right child (false branch)

		// Constructor for leaf node
		Node(double prediction) {
			this.isLeaf = true;
			this.prediction = prediction;
		}

		// Constructor for internal node (numerical split)
		Node(int featureIndex, double threshold) {
			this.featureIndex = featureIndex;
			this.threshold = threshold;
			this.isLeaf = false;
		}

		// Constructor for internal node (categorical split)
		Node(int featureIndex, String category) {
			this.featureIndex = featureIndex;
			this.category = category;
			this.isLeaf = false;
		}
	}

	/**
	 * Create a simple decision tree with specified maximum depth
	 */
	public SimpleDecisionTreeModel(int maxDepth) {
		this.maxDepth = maxDepth;
	}

	/**
	 * Train the decision tree model on provided data
	 * 
	 * @param data   Array of feature vectors
	 * @param labels Array of corresponding labels
	 */
	public void fit(double[][] data, int[] labels) {
		System.out.println("Training simple decision tree model...");
		long startTime = System.currentTimeMillis();

		// Check input validity
		if (data.length != labels.length || data.length == 0) {
			throw new IllegalArgumentException("Data and labels must have the same non-zero length");
		}

		// Build the tree
		root = buildTree(data, labels, 0, new HashSet<>());

		// Calculate accuracy on training data
		int correct = 0;
		for (int i = 0; i < data.length; i++) {
			int predicted = predict(data[i]);
			if (predicted == labels[i]) {
				correct++;
			}
		}
		accuracy = (double) correct / data.length;

		long duration = System.currentTimeMillis() - startTime;
		System.out.println("Training completed in " + duration + "ms");
		System.out.println("Training accuracy: " + accuracy);
	}

	/**
	 * Recursively build the decision tree
	 */
	private Node buildTree(double[][] data, int[] labels, int depth, Set<Integer> usedFeatures) {
		// Base case: maximum depth reached or all samples have the same label
		if (depth >= maxDepth || allSameLabel(labels) || usedFeatures.size() >= data[0].length) {
			return new Node(mostCommonLabel(labels));
		}

		// Find the best feature and threshold for splitting
		int bestFeature = -1;
		double bestThreshold = 0;
		double bestGini = Double.MAX_VALUE;

		for (int feature = 0; feature < data[0].length; feature++) {
			if (usedFeatures.contains(feature)) {
				continue; // Skip already used features
			}

			// Find the mean value for this feature as a simple threshold
			double sum = 0;
			for (double[] datum : data) {
				sum += datum[feature];
			}
			double threshold = sum / data.length;

			// Calculate Gini impurity for this split
			double gini = calculateGiniImpurity(data, labels, feature, threshold);

			if (gini < bestGini) {
				bestGini = gini;
				bestFeature = feature;
				bestThreshold = threshold;
			}
		}

		// If no good split found, create a leaf node
		if (bestFeature == -1) {
			return new Node(mostCommonLabel(labels));
		}

		// Create the node for the best split
		Node node = new Node(bestFeature, bestThreshold);

		// Add the feature to used features set
		usedFeatures.add(bestFeature);

		// Split the data
		List<Integer> leftIndices = new ArrayList<>();
		List<Integer> rightIndices = new ArrayList<>();

		for (int i = 0; i < data.length; i++) {
			if (data[i][bestFeature] <= bestThreshold) {
				leftIndices.add(i);
			} else {
				rightIndices.add(i);
			}
		}

		// Check if the split is empty
		if (leftIndices.isEmpty() || rightIndices.isEmpty()) {
			return new Node(mostCommonLabel(labels));
		}

		// Prepare data for child nodes
		double[][] leftData = new double[leftIndices.size()][data[0].length];
		int[] leftLabels = new int[leftIndices.size()];
		for (int i = 0; i < leftIndices.size(); i++) {
			leftData[i] = data[leftIndices.get(i)];
			leftLabels[i] = labels[leftIndices.get(i)];
		}

		double[][] rightData = new double[rightIndices.size()][data[0].length];
		int[] rightLabels = new int[rightIndices.size()];
		for (int i = 0; i < rightIndices.size(); i++) {
			rightData[i] = data[rightIndices.get(i)];
			rightLabels[i] = labels[rightIndices.get(i)];
		}

		// Create a copy of the used features for each branch
		Set<Integer> leftUsedFeatures = new HashSet<>(usedFeatures);
		Set<Integer> rightUsedFeatures = new HashSet<>(usedFeatures);

		// Build the child nodes
		node.left = buildTree(leftData, leftLabels, depth + 1, leftUsedFeatures);
		node.right = buildTree(rightData, rightLabels, depth + 1, rightUsedFeatures);

		return node;
	}

	/**
	 * Calculate Gini impurity for a potential split
	 */
	private double calculateGiniImpurity(double[][] data, int[] labels, int feature, double threshold) {
		// Count class distributions for left and right splits
		Map<Integer, Integer> leftCounts = new HashMap<>();
		Map<Integer, Integer> rightCounts = new HashMap<>();
		int leftTotal = 0;
		int rightTotal = 0;

		for (int i = 0; i < data.length; i++) {
			int label = labels[i];
			if (data[i][feature] <= threshold) {
				leftCounts.put(label, leftCounts.getOrDefault(label, 0) + 1);
				leftTotal++;
			} else {
				rightCounts.put(label, rightCounts.getOrDefault(label, 0) + 1);
				rightTotal++;
			}
		}

		// Calculate Gini impurity for left node
		double leftGini = 1.0;
		for (int count : leftCounts.values()) {
			double probability = (double) count / leftTotal;
			leftGini -= probability * probability;
		}

		// Calculate Gini impurity for right node
		double rightGini = 1.0;
		for (int count : rightCounts.values()) {
			double probability = (double) count / rightTotal;
			rightGini -= probability * probability;
		}

		// Calculate weighted average Gini impurity
		double weightedGini = 0;
		if (leftTotal > 0) {
			weightedGini += (double) leftTotal / data.length * leftGini;
		}
		if (rightTotal > 0) {
			weightedGini += (double) rightTotal / data.length * rightGini;
		}

		return weightedGini;
	}

	/**
	 * Check if all labels are the same
	 */
	private boolean allSameLabel(int[] labels) {
		if (labels.length == 0) {
			return true;
		}

		int firstLabel = labels[0];
		for (int label : labels) {
			if (label != firstLabel) {
				return false;
			}
		}

		return true;
	}

	/**
	 * Find the most common label in an array
	 */
	private double mostCommonLabel(int[] labels) {
		if (labels.length == 0) {
			return 0;
		}

		Map<Integer, Integer> counts = new HashMap<>();
		for (int label : labels) {
			counts.put(label, counts.getOrDefault(label, 0) + 1);
		}

		int mostCommonLabel = labels[0];
		int maxCount = 0;

		for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
			if (entry.getValue() > maxCount) {
				maxCount = entry.getValue();
				mostCommonLabel = entry.getKey();
			}
		}

		return mostCommonLabel;
	}

	/**
	 * Predict the class for a single sample
	 */
	public int predict(double[] sample) {
		if (root == null) {
			throw new IllegalStateException("Model not trained yet");
		}

		Node node = root;
		while (!node.isLeaf) {
			if (sample[node.featureIndex] <= node.threshold) {
				node = node.left;
			} else {
				node = node.right;
			}
		}

		return (int) node.prediction;
	}

	/**
	 * Calculate accuracy on test data
	 */
	public double evaluate(double[][] testData, int[] testLabels) {
		if (testData.length != testLabels.length || testData.length == 0) {
			throw new IllegalArgumentException("Test data and labels must have the same non-zero length");
		}

		int correct = 0;
		for (int i = 0; i < testData.length; i++) {
			int predicted = predict(testData[i]);
			if (predicted == testLabels[i]) {
				correct++;
			}
		}

		return (double) correct / testData.length;
	}

	/**
	 * Return tree statistics
	 */
	public String getTreeStats() {
		int nodeCount = countNodes(root);
		int depth = calculateDepth(root);

		return String.format("Decision Tree Statistics:\n" + "- Depth: %d (max allowed: %d)\n" + "- Total nodes: %d\n"
				+ "- Training accuracy: %.4f", depth, maxDepth, nodeCount, accuracy);
	}

	/**
	 * Count nodes in the tree
	 */
	private int countNodes(Node node) {
		if (node == null) {
			return 0;
		}
		return 1 + countNodes(node.left) + countNodes(node.right);
	}

	/**
	 * Calculate the actual depth of the tree
	 */
	private int calculateDepth(Node node) {
		if (node == null) {
			return 0;
		}
		return 1 + Math.max(calculateDepth(node.left), calculateDepth(node.right));
	}

	/**
	 * Generate a synthetic dataset for classification
	 */
	public static DataSet generateSyntheticData(int numSamples, int numFeatures, int numClasses, long seed) {
		double[][] data = new double[numSamples][numFeatures];
		int[] labels = new int[numSamples];

		Random random = new Random(seed);

		// Create cluster centers for each class
		double[][] centers = new double[numClasses][numFeatures];
		for (int i = 0; i < numClasses; i++) {
			for (int j = 0; j < numFeatures; j++) {
				centers[i][j] = (random.nextDouble() - 0.5) * 10.0; // Center in range [-5, 5]
			}
		}

		// Generate samples around class centers
		for (int i = 0; i < numSamples; i++) {
			int classLabel = random.nextInt(numClasses);
			labels[i] = classLabel;

			for (int j = 0; j < numFeatures; j++) {
				// Add noise around the class center
				data[i][j] = centers[classLabel][j] + random.nextGaussian();
			}
		}

		return new DataSet(data, labels);
	}

	/**
	 * Class to hold dataset information
	 */
	public static class DataSet {
		public final double[][] data;
		public final int[] labels;

		public DataSet(double[][] data, int[] labels) {
			this.data = data;
			this.labels = labels;
		}

		// Split into training and test sets
		public DataSet[] trainTestSplit(double testRatio, long seed) {
			int testSize = (int) (data.length * testRatio);
			int trainSize = data.length - testSize;

			double[][] trainData = new double[trainSize][data[0].length];
			int[] trainLabels = new int[trainSize];
			double[][] testData = new double[testSize][data[0].length];
			int[] testLabels = new int[testSize];

			// Create a list of indices and shuffle it
			List<Integer> indices = new ArrayList<>();
			for (int i = 0; i < data.length; i++) {
				indices.add(i);
			}
			Collections.shuffle(indices, new Random(seed));

			// Split the data
			for (int i = 0; i < trainSize; i++) {
				int idx = indices.get(i);
				trainData[i] = data[idx];
				trainLabels[i] = labels[idx];
			}

			for (int i = 0; i < testSize; i++) {
				int idx = indices.get(trainSize + i);
				testData[i] = data[idx];
				testLabels[i] = labels[idx];
			}

			return new DataSet[] { new DataSet(trainData, trainLabels), new DataSet(testData, testLabels) };
		}
	}

	public static void main(String[] args) {
		try {
			// Generate synthetic data
			int numSamples = 1000;
			int numFeatures = 5;
			int numClasses = 3;

			System.out.println("Generating synthetic dataset with " + numSamples + " samples, " + numFeatures
					+ " features, and " + numClasses + " classes...");

			DataSet fullDataset = generateSyntheticData(numSamples, numFeatures, numClasses, 42);

			// Split into training and test sets
			DataSet[] splits = fullDataset.trainTestSplit(0.2, 42);
			DataSet trainingSet = splits[0];
			DataSet testSet = splits[1];

			System.out.println("Training set size: " + trainingSet.data.length);
			System.out.println("Test set size: " + testSet.data.length);

			// Create and train the decision tree
			SimpleDecisionTreeModel tree = new SimpleDecisionTreeModel(3); // max depth = 3
			tree.fit(trainingSet.data, trainingSet.labels);

			// Evaluate on test set
			double testAccuracy = tree.evaluate(testSet.data, testSet.labels);
			System.out.println("\n" + tree.getTreeStats());
			System.out.println("Test accuracy: " + testAccuracy);

			// Sample predictions
			System.out.println("\nSample predictions:");
			for (int i = 0; i < 5; i++) {
				int predicted = tree.predict(testSet.data[i]);
				int actual = testSet.labels[i];
				System.out.println("Sample " + i + ": Predicted class = " + predicted + ", Actual class = " + actual);
			}

		} catch (Exception e) {
			System.err.println("Error: " + e.getMessage());
			e.printStackTrace();
		}
	}
}
