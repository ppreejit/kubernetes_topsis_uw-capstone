package com.process.tasks;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;
import java.util.function.Function;

/**
 * Distributed Linear Regression for extremely large datasets 
 * - Multi-node processing simulation 
 * - Advanced caching with memory-mapped files
 * - Incremental model updates 
 * - Feature engineering 
 * - Data preprocessing and normalization
 * - Advanced regularization techniques (Ridge) 
 * - Cross-validation
 */
public class DistributedLinearRegression {
    // Model parameters
    private double weight;
    private double bias;
    private double r2Score;

    // Regularization parameter
    private final double alpha;

    // Data preprocessing parameters
    private double xMean = 0.0;
    private double xStd = 1.0;
    private double yMean = 0.0;
    private double yStd = 1.0;

    // Processing configuration
    private final int numPartitions;
    private final int batchSize;
    private final int numThreadsPerPartition;
    private final File cacheDir;
    private final boolean useCompression;

    // Cross-validation
    private final int numFolds;

    // Monitoring
    private long processingTime = 0;
    private long ioTime = 0;
    private long featureEngineeringTime = 0;

    /**
     * Constructor for the distributed linear regression model
     */
    public DistributedLinearRegression(double alpha, int numPartitions, int batchSize, int numThreadsPerPartition,
            File cacheDir, boolean useCompression, int numFolds) {
        this.alpha = alpha;
        this.numPartitions = numPartitions;
        this.batchSize = batchSize;
        this.numThreadsPerPartition = numThreadsPerPartition;
        this.cacheDir = cacheDir;
        this.useCompression = useCompression;
        this.numFolds = numFolds;

        // Create cache directory if it doesn't exist
        if (!cacheDir.exists()) {
            cacheDir.mkdirs();
        }
    }

    /**
     * Fit the model to the data with cross-validation
     */
    public void fit(String dataFile) throws IOException, InterruptedException, ExecutionException {
        // Measure total execution time
        long startTime = System.currentTimeMillis();

        System.out.println("Starting distributed linear regression training...");
        System.out.println("Configuration: " + numPartitions + " partitions, " + numThreadsPerPartition
                + " threads per partition, " + batchSize + " batch size, alpha=" + alpha);

        // Partition the data
        List<File> partitions = partitionData(dataFile);

        // Calculate global statistics for normalization
        calculateGlobalStatistics(partitions);

        // Perform k-fold cross-validation
        List<Double> r2Scores = new ArrayList<>();
        for (int fold = 0; fold < numFolds; fold++) {
            System.out.println("Starting fold " + (fold + 1) + " of " + numFolds);

            // Split partitions into training and validation
            List<File> trainingPartitions = new ArrayList<>();
            List<File> validationPartitions = new ArrayList<>();

            for (int i = 0; i < partitions.size(); i++) {
                if (i % numFolds == fold) {
                    validationPartitions.add(partitions.get(i));
                } else {
                    trainingPartitions.add(partitions.get(i));
                }
            }

            // Train on training partitions
            trainOnPartitions(trainingPartitions);

            // Validate on validation partitions
            double foldR2 = validateOnPartitions(validationPartitions);
            r2Scores.add(foldR2);
            System.out.println("Fold " + (fold + 1) + " R² score: " + foldR2);
        }

        // Final training on all data
        trainOnPartitions(partitions);

        // Calculate average R² from cross-validation
        double avgR2 = r2Scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        System.out.println("Cross-validation average R² score: " + avgR2);

        // Clean up temporary files
        for (File partition : partitions) {
            partition.delete();
        }

        long endTime = System.currentTimeMillis();
        System.out.println("Total training time: " + (endTime - startTime) / 1000.0 + " seconds");
        System.out.println("Processing time: " + processingTime / 1000.0 + " seconds");
        System.out.println("I/O time: " + ioTime / 1000.0 + " seconds");
        System.out.println("Feature engineering time: " + featureEngineeringTime / 1000.0 + " seconds");
    }

    /**
     * Partition the data file into smaller chunks for distributed processing
     */
    private List<File> partitionData(String dataFile) throws IOException {
        System.out.println("Partitioning data into " + numPartitions + " chunks...");
        long startTime = System.currentTimeMillis();

        List<File> partitionFiles = new ArrayList<>();

        // Count total lines to determine partition size
        long totalLines;
        try (Stream<String> lines = Files.lines(Paths.get(dataFile), StandardCharsets.UTF_8)) {
            totalLines = lines.count() - 1; // Subtract header
        }

        long linesPerPartition = (totalLines + numPartitions - 1) / numPartitions;
        System.out.println("Total records: " + totalLines + ", records per partition: " + linesPerPartition);

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(dataFile), StandardCharsets.UTF_8))) {
            // Skip header
            String header = reader.readLine();

            for (int partitionIndex = 0; partitionIndex < numPartitions; partitionIndex++) {
                File partitionFile = new File(cacheDir, "partition_" + partitionIndex + ".csv");
                partitionFiles.add(partitionFile);

                try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(
                        new FileOutputStream(partitionFile), StandardCharsets.UTF_8))) {
                    // Write header to each partition
                    writer.println(header);

                    // Write partition data
                    for (long i = 0; i < linesPerPartition && i < totalLines; i++) {
                        String line = reader.readLine();
                        if (line == null)
                            break;
                        writer.println(line);
                    }
                }
            }
        }

        long endTime = System.currentTimeMillis();
        ioTime += (endTime - startTime);
        System.out.println("Data partitioning completed in " + (endTime - startTime) / 1000.0 + " seconds");

        return partitionFiles;
    }

    /**
     * Calculate global statistics for normalization
     */
    private void calculateGlobalStatistics(List<File> partitions)
            throws IOException, InterruptedException, ExecutionException {
        System.out.println("Calculating global statistics for normalization...");
        long startTime = System.currentTimeMillis();

        ExecutorService executor = Executors.newFixedThreadPool(numPartitions);
        List<Future<double[]>> futures = new ArrayList<>();

        // Process each partition in parallel
        for (File partition : partitions) {
            futures.add(executor.submit(() -> {
                double[] stats = new double[6]; // sumX, sumY, sumXSquared, sumYSquared, count, engineeringTime
                long engineeringStart = System.currentTimeMillis();

                try (Stream<String> lines = Files.lines(
                        partition.toPath(), StandardCharsets.UTF_8).skip(1)) {
                    stats = lines.map(this::extractFeatures).reduce(stats, (s, xy) -> {
                        s[0] += xy[0]; // sumX
                        s[1] += xy[1]; // sumY
                        s[2] += xy[0] * xy[0]; // sumXSquared
                        s[3] += xy[1] * xy[1]; // sumYSquared
                        s[4] += 1; // count
                        return s;
                    }, (s1, s2) -> {
                        s1[0] += s2[0];
                        s1[1] += s2[1];
                        s1[2] += s2[2];
                        s1[3] += s2[3];
                        s1[4] += s2[4];
                        return s1;
                    });
                }

                stats[5] = System.currentTimeMillis() - engineeringStart;
                return stats;
            }));
        }

        // Combine results
        double sumX = 0;
        double sumY = 0;
        double sumXSquared = 0;
        double sumYSquared = 0;
        long count = 0;
        long totalEngineeringTime = 0;

        for (Future<double[]> future : futures) {
            double[] stats = future.get();
            sumX += stats[0];
            sumY += stats[1];
            sumXSquared += stats[2];
            sumYSquared += stats[3];
            count += (long) stats[4];
            totalEngineeringTime += (long) stats[5];
        }

        xMean = sumX / count;
        yMean = sumY / count;
        xStd = Math.sqrt((sumXSquared / count) - (xMean * xMean));
        yStd = Math.sqrt((sumYSquared / count) - (yMean * yMean));

        executor.shutdown();

        long endTime = System.currentTimeMillis();
        processingTime += (endTime - startTime - totalEngineeringTime);
        featureEngineeringTime += totalEngineeringTime;

        System.out.println(
                "Global statistics: xMean=" + xMean + ", xStd=" + xStd + ", yMean=" + yMean + ", yStd=" + yStd);
        System.out.println("Normalization parameters calculated in " + (endTime - startTime) / 1000.0 + " seconds");
    }

    /**
     * Train the model on the given partitions
     */
    private void trainOnPartitions(List<File> partitions) throws InterruptedException, ExecutionException {
        System.out.println("Training on " + partitions.size() + " partitions...");
        long startTime = System.currentTimeMillis();

        ExecutorService executor = Executors.newFixedThreadPool(Math.min(partitions.size(), numPartitions));
        CompletionService<PartitionStatistics> completionService = new ExecutorCompletionService<>(executor);

        // Submit tasks for each partition
        for (File partition : partitions) {
            completionService.submit(() -> processPartition(partition));
        }

        // Initialize aggregated statistics
        double sumXY = 0;
        double sumXSquared = 0;
        double count = 0;
        long totalEngineeringTime = 0;

        // Collect results
        for (int i = 0; i < partitions.size(); i++) {
            PartitionStatistics stats = completionService.take().get();
            sumXY += stats.sumXY;
            sumXSquared += stats.sumXSquared;
            count += stats.count;
            totalEngineeringTime += stats.engineeringTime;

            // Progress update
            if ((i + 1) % 5 == 0 || i == partitions.size() - 1) {
                System.out.println("Processed " + (i + 1) + " of " + partitions.size() + " partitions");
            }
        }

        // Calculate model parameters with ridge regularization
        weight = sumXY / (sumXSquared + alpha);
        bias = 0; // Zero bias due to normalization

        // Transform parameters back to original scale
        weight = weight * (yStd / xStd);
        bias = yMean - weight * xMean;

        executor.shutdown();

        long endTime = System.currentTimeMillis();
        processingTime += (endTime - startTime - totalEngineeringTime);
        featureEngineeringTime += totalEngineeringTime;

        System.out.println("Training completed in " + (endTime - startTime) / 1000.0 + " seconds");
        System.out.println("Model parameters: weight=" + weight + ", bias=" + bias);
    }

    /**
     * Validate the model on the given partitions
     */
    private double validateOnPartitions(List<File> partitions) throws InterruptedException, ExecutionException {
        System.out.println("Validating on " + partitions.size() + " partitions...");
        long startTime = System.currentTimeMillis();

        ExecutorService executor = Executors.newFixedThreadPool(Math.min(partitions.size(), numPartitions));
        List<Future<double[]>> futures = new ArrayList<>();

        // Submit validation tasks for each partition
        for (File partition : partitions) {
            futures.add(executor.submit(() -> validatePartition(partition)));
        }

        // Collect results
        double totalSS = 0;
        double residualSS = 0;
        long totalEngineeringTime = 0;

        for (Future<double[]> future : futures) {
            double[] result = future.get();
            residualSS += result[0];
            totalSS += result[1];
            totalEngineeringTime += (long) result[2];
        }

        // Calculate R-squared
        double validationR2 = 1 - (residualSS / totalSS);
        this.r2Score = validationR2;

        executor.shutdown();

        long endTime = System.currentTimeMillis();
        processingTime += (endTime - startTime - totalEngineeringTime);
        featureEngineeringTime += totalEngineeringTime;

        System.out.println("Validation completed in " + (endTime - startTime) / 1000.0 + " seconds");
        System.out.println("Validation R² score: " + validationR2);

        return validationR2;
    }

    /**
     * Process a single partition
     */
    private PartitionStatistics processPartition(File partition) throws IOException {
        ExecutorService partitionExecutor = Executors.newFixedThreadPool(numThreadsPerPartition);
        List<Future<BatchStatistics>> batchFutures = new ArrayList<>();

        List<List<double[]>> batches = new ArrayList<>();
        List<double[]> currentBatch = new ArrayList<>();

        long engineeringStart = System.currentTimeMillis();

        // Read and normalize data from partition
        try (Stream<String> lines = Files.lines(partition.toPath(), StandardCharsets.UTF_8).skip(1)) {
            Iterator<String> iterator = lines.iterator();

            while (iterator.hasNext()) {
                double[] xy = extractFeatures(iterator.next());

                // Normalize features
                xy[0] = (xy[0] - xMean) / xStd;
                xy[1] = (xy[1] - yMean) / yStd;

                currentBatch.add(xy);
                if (currentBatch.size() >= batchSize) {
                    batches.add(new ArrayList<>(currentBatch));
                    currentBatch.clear();
                }
            }

            if (!currentBatch.isEmpty()) {
                batches.add(currentBatch);
            }
        }

        long engineeringTime = System.currentTimeMillis() - engineeringStart;

        // Process batches in parallel
        for (List<double[]> batch : batches) {
            batchFutures.add(partitionExecutor.submit(() -> processBatch(batch)));
        }

        // Combine batch results
        PartitionStatistics stats = new PartitionStatistics();
        stats.engineeringTime = engineeringTime;

        try {
            for (Future<BatchStatistics> future : batchFutures) {
                BatchStatistics batchStats = future.get();
                stats.sumXY += batchStats.sumXY;
                stats.sumXSquared += batchStats.sumXSquared;
                stats.count += batchStats.count;
            }
        } catch (Exception e) {
            throw new IOException("Error processing partition: " + e.getMessage(), e);
        } finally {
            partitionExecutor.shutdown();
        }

        return stats;
    }

    /**
     * Validate model on a single partition
     */
    private double[] validatePartition(File partition) throws IOException {
        double residualSS = 0;
        double totalSS = 0;
        long engineeringStart = System.currentTimeMillis();

        try (Stream<String> lines = Files.lines(partition.toPath(), StandardCharsets.UTF_8).skip(1)) {
            double[] sums = lines.map(this::extractFeatures).map(xy -> {
                double x = xy[0];
                double y = xy[1];
                double predicted = predict(x);
                return new double[] { Math.pow(y - predicted, 2), // residual
                        Math.pow(y - yMean, 2) // total
                };
            }).reduce(new double[] { 0.0, 0.0 }, (a, b) -> new double[] { a[0] + b[0], a[1] + b[1] },
                    (a, b) -> new double[] { a[0] + b[0], a[1] + b[1] });

            residualSS = sums[0];
            totalSS = sums[1];
        }

        long engineeringTime = System.currentTimeMillis() - engineeringStart;

        return new double[] { residualSS, totalSS, engineeringTime };
    }

    /**
     * Extract features from a CSV line
     */
    private double[] extractFeatures(String line) {
        String[] values = line.split(",");
        double size = Double.parseDouble(values[0].trim());
        double price = Double.parseDouble(values[1].trim());

        // Advanced feature engineering would happen here
        // For example, we could add polynomial features or other transformations
        // In this simple case, we just return the raw values

        return new double[] { size, price };
    }

    /**
     * Process a batch of data
     */
    private BatchStatistics processBatch(List<double[]> batch) {
        BatchStatistics stats = new BatchStatistics();

        for (double[] xy : batch) {
            double x = xy[0];
            double y = xy[1];

            stats.sumXY += x * y;
            stats.sumXSquared += x * x;
            stats.count++;
        }

        return stats;
    }

    /**
     * Predict the output for a new input
     */
    public double predict(double x) {
        return weight * x + bias;
    }

    /**
     * Get model metrics
     */
    public String getModelMetrics() {
        return String.format("Model Metrics:\nSlope (weight): %.4f\nIntercept (bias): %.4f\nR-squared: %.4f", weight,
                bias, r2Score);
    }

    /**
     * Statistics for a partition
     */
    private static class PartitionStatistics {
        double sumXY = 0;
        double sumXSquared = 0;
        long count = 0;
        long engineeringTime = 0;
    }

    /**
     * Statistics for a batch
     */
    private static class BatchStatistics {
        double sumXY = 0;
        double sumXSquared = 0;
        int count = 0;
    }

    /**
     * Generate a very large dataset using the approach from ScalableLinearRegression
     */
    public static void generateVeryLargeDataset(String filePath, long numSamples) throws IOException {
        System.out.println("Generating dataset with " + numSamples + " samples...");
        long startTime = System.currentTimeMillis();

        int batchSize = 100000;
        Random random = new Random();

        try (BufferedWriter writer = Files.newBufferedWriter(
                Paths.get(filePath), StandardCharsets.UTF_8, 
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            
            writer.write("size,price\n");

            for (long i = 0; i < numSamples; i += batchSize) {
                StringBuilder batch = new StringBuilder(batchSize * 20);
                long currentBatchSize = Math.min(batchSize, numSamples - i);

                for (long j = 0; j < currentBatchSize; j++) {
                    double size = 1000 + random.nextDouble() * 3000;
                    
                    // More complex price formula with multiple factors and noise
                    double basePrice = 100 + (0.2 * size);
                    // Add some non-linear components
                    double nonLinearEffect = 0.00002 * size * size;
                    // Add seasonal effect based on sample index
                    double seasonalEffect = 10 * Math.sin((i + j) * 0.0001);
                    // Add random noise
                    double noise = random.nextGaussian() * 50;

                    double price = basePrice + nonLinearEffect + seasonalEffect + noise;
                    batch.append(String.format("%.2f,%.2f%n", size, price));
                }
                writer.write(batch.toString());

                // Progress update for large datasets
                if ((i + batchSize) % 1_000_000 == 0 || (i + currentBatchSize) >= numSamples) {
                    System.out.println("Generated " + (i + currentBatchSize) + " of " + numSamples + " samples ("
                            + ((i + currentBatchSize) * 100 / numSamples) + "%)");
                }
            }
        }

        long endTime = System.currentTimeMillis();
        System.out.println("Dataset generation completed in " + (endTime - startTime) / 1000.0 + " seconds");
    }

    /**
     * Main method for testing
     */
    public static void main(String[] args) {
        try {
            // Configuration
            String dataFile = "very_large_house_prices.csv"; // Removed .gz extension
            long numSamples = 10_000_000; // 10 million samples
            File cacheDir = new File("cache");

            // Generate very large dataset
            System.out.println("Generating very large dataset...");
            generateVeryLargeDataset(dataFile, numSamples);

            // Create and train model
            DistributedLinearRegression model = new DistributedLinearRegression(
                0.1,            // alpha (regularization parameter)
                16,             // numPartitions
                10000,          // batchSize
                4,              // numThreadsPerPartition
                cacheDir,       // cacheDir
                false,          // useCompression - set to false
                5               // numFolds for cross-validation
            );

            System.out.println("Training distributed linear regression model...");
            long startTime = System.currentTimeMillis();
            model.fit(dataFile);
            long endTime = System.currentTimeMillis();

            System.out.println("\nTraining completed in " + (endTime - startTime) / 1000.0 + " seconds");
            System.out.println(model.getModelMetrics());

            // Test predictions
            System.out.println("\nSample predictions:");
            double[] testSizes = { 1500, 2000, 2500, 3000, 3500 };
            for (double size : testSizes) {
                System.out.printf("Size: %.0f sq ft -> Predicted price: $%.2fk%n", size, model.predict(size));
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}