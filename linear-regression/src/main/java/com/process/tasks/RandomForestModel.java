package com.process.tasks;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

/**
 * Random Forest implementation for medium-sized datasets
 * - Ensemble of decision trees with bootstrapping
 * - Feature subspace sampling
 * - Parallel tree training
 * - Handles large datasets through batch processing
 */
public class RandomForestModel {
    private final int numTrees;
    private final int maxDepth;
    private final int maxFeatures;
    private final int minSamplesPerLeaf;
    private final int numThreads;
    private final int batchSize;
    private List<DecisionTree> trees;
    private double oobScore;
    
    // Performance metrics
    private long trainingTime = 0;
    private long predictionTime = 0;
    
    /**
     * Create a Random Forest model with specified parameters
     * 
     * @param numTrees Number of trees in the forest
     * @param maxDepth Maximum depth of each tree
     * @param maxFeatures Maximum number of features to consider for each split
     * @param minSamplesPerLeaf Minimum number of samples required to be a leaf node
     * @param numThreads Number of threads to use for parallel training
     * @param batchSize Batch size for processing large datasets
     */
    public RandomForestModel(int numTrees, int maxDepth, int maxFeatures, int minSamplesPerLeaf, 
                           int numThreads, int batchSize) {
        this.numTrees = numTrees;
        this.maxDepth = maxDepth;
        this.maxFeatures = maxFeatures;
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.numThreads = numThreads;
        this.batchSize = batchSize;
        this.trees = new ArrayList<>(numTrees);
    }
    
    /**
     * Train the Random Forest model on the provided data
     */
    public void fit(double[][] data, int[] labels) {
        System.out.println("Training Random Forest with " + numTrees + " trees...");
        System.out.println("Configuration: maxDepth=" + maxDepth + ", maxFeatures=" + maxFeatures + 
                         ", minSamplesPerLeaf=" + minSamplesPerLeaf + ", threads=" + numThreads);
        
        long startTime = System.currentTimeMillis();
        
        // Initialize out-of-bag predictions for OOB score calculation
        int[] oobPredictions = new int[data.length];
        int[] oobCounts = new int[data.length];
        
        // Train trees in parallel
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<TreeTrainingResult>> futures = new ArrayList<>();
        
        for (int t = 0; t < numTrees; t++) {
            final int treeIndex = t;
            futures.add(executor.submit(() -> {
                Random random = new Random(treeIndex);  // Different seed for each tree
                
                // Bootstrap sampling
                int[] indices = new int[data.length];
                boolean[] inBag = new boolean[data.length];
                for (int i = 0; i < data.length; i++) {
                    indices[i] = random.nextInt(data.length);
                    inBag[indices[i]] = true;
                }
                
                // Create bootstrapped dataset
                List<double[]> bootstrapData = new ArrayList<>();
                List<Integer> bootstrapLabels = new ArrayList<>();
                
                for (int idx : indices) {
                    bootstrapData.add(data[idx]);
                    bootstrapLabels.add(labels[idx]);
                }
                
                // Train tree on bootstrapped data
                DecisionTree tree = new DecisionTree(maxDepth, maxFeatures, minSamplesPerLeaf, random.nextLong());
                tree.fit(bootstrapData.toArray(new double[0][]), bootstrapLabels.stream().mapToInt(i -> i).toArray());
                
                // Make out-of-bag predictions
                Map<Integer, Integer> oobPreds = new HashMap<>();
                for (int i = 0; i < data.length; i++) {
                    if (!inBag[i]) {
                        int prediction = tree.predict(data[i]);
                        oobPreds.put(i, prediction);
                    }
                }
                
                return new TreeTrainingResult(tree, oobPreds, treeIndex);
            }));
        }
        
        // Collect trained trees and update OOB predictions
        trees.clear();
        for (int t = 0; t < numTrees; t++) {
            try {
                TreeTrainingResult result = futures.get(t).get();
                trees.add(result.tree);
                
                // Update OOB predictions and counts
                for (Map.Entry<Integer, Integer> entry : result.oobPredictions.entrySet()) {
                    int index = entry.getKey();
                    int prediction = entry.getValue();
                    
                    synchronized (this) {
                        if (oobCounts[index] == 0) {
                            oobPredictions[index] = prediction;
                        } else {
                            // Simple majority voting
                            oobPredictions[index] = (oobPredictions[index] * oobCounts[index] + prediction) / (oobCounts[index] + 1);
                        }
                        oobCounts[index]++;
                    }
                }
                
                // Progress update
                if ((t + 1) % 5 == 0 || t == numTrees - 1) {
                    System.out.println("Trained " + (t + 1) + " of " + numTrees + " trees");
                }
                
            } catch (Exception e) {
                System.err.println("Error training tree " + t + ": " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        executor.shutdown();
        
        // Calculate OOB score
        int oobCorrect = 0;
        int oobTotal = 0;
        for (int i = 0; i < data.length; i++) {
            if (oobCounts[i] > 0) {
                if (oobPredictions[i] == labels[i]) {
                    oobCorrect++;
                }
                oobTotal++;
            }
        }
        oobScore = (double) oobCorrect / oobTotal;
        
        trainingTime = System.currentTimeMillis() - startTime;
        System.out.println("Training completed in " + trainingTime / 1000.0 + " seconds");
        System.out.println("Out-of-bag score: " + oobScore);
    }
    
    /**
     * Predict the class label for a single sample
     */
    public int predict(double[] sample) {
        if (trees.isEmpty()) {
            throw new IllegalStateException("Model not trained yet");
        }
        
        long startTime = System.currentTimeMillis();
        
        // Voting from all trees
        Map<Integer, Integer> votes = new HashMap<>();
        
        for (DecisionTree tree : trees) {
            int prediction = tree.predict(sample);
            votes.put(prediction, votes.getOrDefault(prediction, 0) + 1);
        }
        
        // Find the class with most votes
        int predictedClass = -1;
        int maxVotes = 0;
        for (Map.Entry<Integer, Integer> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                predictedClass = entry.getKey();
            }
        }
        
        predictionTime += System.currentTimeMillis() - startTime;
        return predictedClass;
    }
    
    /**
     * Predict class labels for multiple samples
     */
    public int[] predict(double[][] samples) {
        int[] predictions = new int[samples.length];
        
        for (int i = 0; i < samples.length; i++) {
            predictions[i] = predict(samples[i]);
        }
        
        return predictions;
    }
    
    /**
     * Calculate accuracy on test data
     */
    public double evaluate(double[][] testData, int[] testLabels) {
        if (testData.length != testLabels.length || testData.length == 0) {
            throw new IllegalArgumentException("Test data and labels must have the same non-zero length");
        }
        
        int[] predictions = predict(testData);
        
        int correct = 0;
        for (int i = 0; i < testLabels.length; i++) {
            if (predictions[i] == testLabels[i]) {
                correct++;
            }
        }
        
        return (double) correct / testLabels.length;
    }
    
    /**
     * Return model statistics
     */
    public String getModelStats() {
        if (trees.isEmpty()) {
            return "Model not trained yet";
        }
        
        int totalNodes = 0;
        int totalDepth = 0;
        
        for (DecisionTree tree : trees) {
            totalNodes += tree.getNodeCount();
            totalDepth += tree.getDepth();
        }
        
        double avgNodes = (double) totalNodes / trees.size();
        double avgDepth = (double) totalDepth / trees.size();
        
        return String.format("Random Forest Statistics:\n" +
                "- Number of trees: %d\n" +
                "- Average tree depth: %.2f\n" +
                "- Average nodes per tree: %.2f\n" +
                "- Out-of-bag score: %.4f\n" +
                "- Training time: %.2f seconds\n" +
                "- Average prediction time: %.2f ms",
                trees.size(), avgDepth, avgNodes, oobScore, 
                trainingTime / 1000.0, predictionTime / (double) (trees.size() * trees.get(0).getPredictionCount()));
    }
    
    /**
     * Calculate feature importances
     */
    public double[] getFeatureImportances(int numFeatures) {
        if (trees.isEmpty()) {
            throw new IllegalStateException("Model not trained yet");
        }
        
        double[] importances = new double[numFeatures];
        
        for (DecisionTree tree : trees) {
            double[] treeImportances = tree.getFeatureImportances(numFeatures);
            for (int i = 0; i < numFeatures; i++) {
                importances[i] += treeImportances[i] / trees.size();
            }
        }
        
        return importances;
    }
    
    /**
     * Helper class to hold tree training results
     */
    private static class TreeTrainingResult {
        DecisionTree tree;
        Map<Integer, Integer> oobPredictions;
        int treeIndex;
        
        TreeTrainingResult(DecisionTree tree, Map<Integer, Integer> oobPredictions, int treeIndex) {
            this.tree = tree;
            this.oobPredictions = oobPredictions;
            this.treeIndex = treeIndex;
        }
    }
    
    /**
     * Generate a larger synthetic dataset for classification
     */
    public static DataSet generateSyntheticData(int numSamples, int numFeatures, int numClasses, long seed) {
        double[][] data = new double[numSamples][numFeatures];
        int[] labels = new int[numSamples];
        
        Random random = new Random(seed);
        
        // Create multiple cluster centers for each class for more complex decision boundaries
        int centersPerClass = 3;
        double[][][] centers = new double[numClasses][centersPerClass][numFeatures];
        
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < centersPerClass; j++) {
                for (int k = 0; k < numFeatures; k++) {
                    centers[i][j][k] = (random.nextDouble() - 0.5) * 20.0;  // Centers in range [-10, 10]
                }
            }
        }
        
        // Generate samples with more complex distribution
        for (int i = 0; i < numSamples; i++) {
            int classLabel = random.nextInt(numClasses);
            labels[i] = classLabel;
            
            // Select one of the centers for this class
            int centerIdx = random.nextInt(centersPerClass);
            
            for (int j = 0; j < numFeatures; j++) {
                // Add noise around the class center
                double noise = random.nextGaussian() * 2.0;  // More noise for harder classification
                data[i][j] = centers[classLabel][centerIdx][j] + noise;
                
                // Add some non-linear feature interactions for more complexity
                if (j > 0 && random.nextDouble() < 0.3) {
                    data[i][j] += 0.5 * data[i][j-1] * noise;
                }
            }
            
            // Add some outliers
            if (random.nextDouble() < 0.01) {
                for (int j = 0; j < numFeatures; j++) {
                    data[i][j] += (random.nextDouble() - 0.5) * 30.0;
                }
            }
        }
        
        return new DataSet(data, labels);
    }
    
    /**
     * Save dataset to file
     */
    public static void saveDatasetToFile(DataSet dataset, String filePath) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(filePath))) {
            // Write header
            for (int i = 0; i < dataset.data[0].length; i++) {
                writer.write("feature_" + i + ",");
            }
            writer.write("label\n");
            
            // Write data
            for (int i = 0; i < dataset.data.length; i++) {
                for (int j = 0; j < dataset.data[i].length; j++) {
                    writer.write(String.format("%.6f,", dataset.data[i][j]));
                }
                writer.write(Integer.toString(dataset.labels[i]));
                writer.write("\n");
            }
        }
    }
    
    /**
     * Load dataset from file
     */
    public static DataSet loadDatasetFromFile(String filePath) throws IOException {
        List<double[]> data = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(filePath))) {
            String header = reader.readLine();  // Skip header
            String line;
            
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                double[] features = new double[values.length - 1];
                
                for (int i = 0; i < values.length - 1; i++) {
                    features[i] = Double.parseDouble(values[i]);
                }
                
                int label = Integer.parseInt(values[values.length - 1]);
                
                data.add(features);
                labels.add(label);
            }
        }
        
        // Convert lists to arrays
        double[][] dataArray = data.toArray(new double[0][]);
        int[] labelsArray = labels.stream().mapToInt(i -> i).toArray();
        
        return new DataSet(dataArray, labelsArray);
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
            
            return new DataSet[] { 
                new DataSet(trainData, trainLabels), 
                new DataSet(testData, testLabels) 
            };
        }
    }
    
    /**
     * Decision Tree implementation for Random Forest
     */
    private static class DecisionTree {
        private Node root;
        private final int maxDepth;
        private final int maxFeatures;
        private final int minSamplesPerLeaf;
        private final Random random;
        private final long seed;
        private int predictionCount = 0;
        private Map<Integer, Double> featureImportances = new HashMap<>();
        
        // Decision tree node
        private static class Node {
            int featureIndex;
            double threshold;
            boolean isLeaf;
            int prediction;
            Node left;
            Node right;
            double impurityDecrease;
            
            // Constructor for leaf node
            Node(int prediction) {
                this.isLeaf = true;
                this.prediction = prediction;
            }
            
            // Constructor for internal node
            Node(int featureIndex, double threshold) {
                this.featureIndex = featureIndex;
                this.threshold = threshold;
                this.isLeaf = false;
            }
        }
        
        /**
         * Create a decision tree with specified parameters
         */
        public DecisionTree(int maxDepth, int maxFeatures, int minSamplesPerLeaf, long seed) {
            this.maxDepth = maxDepth;
            this.maxFeatures = maxFeatures;
            this.minSamplesPerLeaf = minSamplesPerLeaf;
            this.seed = seed;
            this.random = new Random(seed);
        }
        
        /**
         * Train the decision tree on provided data
         */
        public void fit(double[][] data, int[] labels) {
            featureImportances.clear();
            root = buildTree(data, labels, 0);
        }
        
        /**
         * Recursively build the decision tree
         */
        private Node buildTree(double[][] data, int[] labels, int depth) {
            // Check stopping criteria
            if (depth >= maxDepth || data.length < minSamplesPerLeaf * 2 || allSameLabel(labels)) {
                return new Node(mostCommonLabel(labels));
            }
            
            // Select a random subset of features to consider
            int numFeaturesToConsider = Math.min(maxFeatures, data[0].length);
            int[] featureIndices = new int[numFeaturesToConsider];
            boolean[] selectedFeatures = new boolean[data[0].length];
            
            int count = 0;
            while (count < numFeaturesToConsider) {
                int featureIdx = random.nextInt(data[0].length);
                if (!selectedFeatures[featureIdx]) {
                    featureIndices[count++] = featureIdx;
                    selectedFeatures[featureIdx] = true;
                }
            }
            
            // Find the best split
            int bestFeature = -1;
            double bestThreshold = 0;
            double bestGini = Double.MAX_VALUE;
            double bestImpurityDecrease = 0;
            
            for (int featureIdx : featureIndices) {
                // Sort data by this feature
                Integer[] sortedIndices = new Integer[data.length];
                for (int i = 0; i < data.length; i++) {
                    sortedIndices[i] = i;
                }
                final int feature = featureIdx;
                Arrays.sort(sortedIndices, Comparator.comparingDouble(i -> data[i][feature]));
                
                // Try different thresholds
                for (int i = 0; i < data.length - 1; i++) {
                    // Skip if values are the same
                    if (data[sortedIndices[i]][featureIdx] == data[sortedIndices[i+1]][featureIdx]) {
                        continue;
                    }
                    
                    // Calculate threshold as midpoint
                    double threshold = (data[sortedIndices[i]][featureIdx] + data[sortedIndices[i+1]][featureIdx]) / 2.0;
                    
                    // Calculate Gini impurity for this split
                    double[] giniInfo = calculateGiniImpurity(data, labels, featureIdx, threshold);
                    double gini = giniInfo[0];
                    double impurityDecrease = giniInfo[1];
                    
                    if (gini < bestGini) {
                        bestGini = gini;
                        bestFeature = featureIdx;
                        bestThreshold = threshold;
                        bestImpurityDecrease = impurityDecrease;
                    }
                }
            }
            
            // If no good split found, create leaf node
            if (bestFeature == -1) {
                return new Node(mostCommonLabel(labels));
            }
            
            // Create the node for the best split
            Node node = new Node(bestFeature, bestThreshold);
            node.impurityDecrease = bestImpurityDecrease;
            
            // Update feature importance
            featureImportances.put(bestFeature, featureImportances.getOrDefault(bestFeature, 0.0) + bestImpurityDecrease);
            
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
            
            // Check if the split is valid
            if (leftIndices.size() < minSamplesPerLeaf || rightIndices.size() < minSamplesPerLeaf) {
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
            
            // Build child nodes
            node.left = buildTree(leftData, leftLabels, depth + 1);
            node.right = buildTree(rightData, rightLabels, depth + 1);
            
            return node;
        }
        
        /**
         * Calculate Gini impurity for a potential split and impurity decrease
         */
        private double[] calculateGiniImpurity(double[][] data, int[] labels, int feature, double threshold) {
            Map<Integer, Integer> leftCounts = new HashMap<>();
            Map<Integer, Integer> rightCounts = new HashMap<>();
            int leftTotal = 0;
            int rightTotal = 0;
            
            // Calculate parent Gini
            Map<Integer, Integer> parentCounts = new HashMap<>();
            for (int label : labels) {
                parentCounts.put(label, parentCounts.getOrDefault(label, 0) + 1);
            }
            
            double parentGini = 1.0;
            for (int count : parentCounts.values()) {
                double probability = (double) count / labels.length;
                parentGini -= probability * probability;
            }
            
            // Calculate left and right counts
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
            
            // Calculate impurity decrease
            double impurityDecrease = parentGini - weightedGini;
            
            return new double[] { weightedGini, impurityDecrease };
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
        private int mostCommonLabel(int[] labels) {
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
            
            predictionCount++;
            
            Node node = root;
            while (!node.isLeaf) {
                if (sample[node.featureIndex] <= node.threshold) {
                    node = node.left;
                } else {
                    node = node.right;
                }
            }
            
            return node.prediction;
        }
        
        /**
         * Get tree statistics
         */
        public int getNodeCount() {
            return countNodes(root);
        }
        
        public int getDepth() {
            return calculateDepth(root);
        }
        
        public int getPredictionCount() {
            return predictionCount;
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
         * Get feature importances
         */
        public double[] getFeatureImportances(int numFeatures) {
            double[] importances = new double[numFeatures];
            double total = 0.0;
            
            // Sum up all importance values
            for (Map.Entry<Integer, Double> entry : featureImportances.entrySet()) {
                total += entry.getValue();
            }
            
            // Normalize importances
            if (total > 0) {
                for (Map.Entry<Integer, Double> entry : featureImportances.entrySet()) {
                    importances[entry.getKey()] = entry.getValue() / total;
                }
            }
            
            return importances;
        }
    }
    
    public static void main(String[] args) {
        try {
            // Generate synthetic data
            int numSamples = 100_000;
            int numFeatures = 10;
            int numClasses = 5;
            
            System.out.println("Generating synthetic dataset with " + numSamples + " samples, " + 
                              numFeatures + " features, and " + numClasses + " classes...");
            
            DataSet fullDataset = generateSyntheticData(numSamples, numFeatures, numClasses, 42);
            
            // Split into training and test sets
            DataSet[] splits = fullDataset.trainTestSplit(0.2, 42);
            DataSet trainingSet = splits[0];
            DataSet testSet = splits[1];
            
            System.out.println("Training set size: " + trainingSet.data.length);
            System.out.println("Test set size: " + testSet.data.length);
            
            // Create and train the Random Forest model
            int numThreads = Runtime.getRuntime().availableProcessors();
            RandomForestModel model = new RandomForestModel(
                15,                   // numTrees
                8,                    // maxDepth
                (int)Math.sqrt(numFeatures), // maxFeatures
                5,                    // minSamplesPerLeaf
                numThreads,           // numThreads
                10000                 // batchSize
            );
            
            // Train the model
            long startTime = System.currentTimeMillis();
            model.fit(trainingSet.data, trainingSet.labels);
            long endTime = System.currentTimeMillis();
            
            System.out.println("Training completed in " + (endTime - startTime) / 1000.0 + " seconds");
            
            // Evaluate on test set
            System.out.println("Evaluating on test set...");
            double testAccuracy = model.evaluate(testSet.data, testSet.labels);
            System.out.println("Test accuracy: " + testAccuracy);
            
            // Print model statistics
            System.out.println("\n" + model.getModelStats());
            
            // Print feature importances
            double[] importances = model.getFeatureImportances(numFeatures);
            System.out.println("\nFeature importances:");
            for (int i = 0; i < importances.length; i++) {
                System.out.printf("Feature %d: %.4f%n", i, importances[i]);
            }
            
            // Sample predictions
            System.out.println("\nSample predictions:");
            for (int i = 0; i < 5; i++) {
                int predicted = model.predict(testSet.data[i]);
                int actual = testSet.labels[i];
                System.out.println("Sample " + i + ": Predicted class = " + predicted + 
                                  ", Actual class = " + actual);
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}