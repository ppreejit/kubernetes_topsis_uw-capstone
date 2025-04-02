package com.process.tasks;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LinearRegressionSmallDataset {
    private double weight; // slope
    private double bias; // intercept
    private double r2Score; // R-squared value for model evaluation

    // Train the model
    public void fit(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("Input arrays must have the same length");
        }
        if (x.length == 0) {
            throw new IllegalArgumentException("Input arrays cannot be empty");
        }

        double xMean = Arrays.stream(x).average().orElse(0.0);
        double yMean = Arrays.stream(y).average().orElse(0.0);

        // Calculate the weight (slope)
        double numerator = 0;
        double denominator = 0;
        for (int i = 0; i < x.length; i++) {
            numerator += (x[i] - xMean) * (y[i] - yMean);
            denominator += Math.pow(x[i] - xMean, 2);
        }
        weight = numerator / denominator;

        // Calculate the bias (intercept)
        bias = yMean - weight * xMean;

        // Calculate R-squared
        calculateR2Score(x, y, yMean);
    }

    // Calculate R-squared score
    private void calculateR2Score(double[] x, double[] y, double yMean) {
        double totalSS = 0;
        double residualSS = 0;
        
        for (int i = 0; i < x.length; i++) {
            double predicted = predict(x[i]);
            residualSS += Math.pow(y[i] - predicted, 2);
            totalSS += Math.pow(y[i] - yMean, 2);
        }
        
        r2Score = 1 - (residualSS / totalSS);
    }

    // Predict the output for a new input
    public double predict(double x) {
        return weight * x + bias;
    }

    // Get model metrics
    public String getModelMetrics() {
        return String.format("Model Metrics:\nSlope (weight): %.4f\nIntercept (bias): %.4f\nR-squared: %.4f", 
                           weight, bias, r2Score);
    }

    // Load dataset from CSV with validation
    public static double[][] loadDataset(String filePath) throws IOException {
        List<Double> sizes = new ArrayList<>();
        List<Double> prices = new ArrayList<>();
        
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine(); // Skip header
            if (line == null) {
                throw new IOException("Empty file");
            }
            
            int lineNumber = 1;
            while ((line = br.readLine()) != null) {
                lineNumber++;
                try {
                    String[] values = line.split(",");
                    if (values.length != 2) {
                        System.err.println("Warning: Invalid data format at line " + lineNumber + ", skipping...");
                        continue;
                    }
                    
                    double size = Double.parseDouble(values[0].trim());
                    double price = Double.parseDouble(values[1].trim());
                    
                    // Basic data validation
                    if (size <= 0 || price <= 0) {
                        System.err.println("Warning: Invalid values at line " + lineNumber + ", skipping...");
                        continue;
                    }
                    
                    sizes.add(size);
                    prices.add(price);
                } catch (NumberFormatException e) {
                    System.err.println("Warning: Invalid number format at line " + lineNumber + ", skipping...");
                }
            }
        }
        
        if (sizes.isEmpty()) {
            throw new IOException("No valid data found in the file");
        }

        double[] sizeArray = sizes.stream().mapToDouble(Double::doubleValue).toArray();
        double[] priceArray = prices.stream().mapToDouble(Double::doubleValue).toArray();
        return new double[][] { sizeArray, priceArray };
    }

    // Generate sample dataset
    public static void generateSampleDataset(String filePath, int numSamples) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            writer.println("size,price");
            Random random = new Random();
            
            // Generate realistic house data with some noise
            for (int i = 0; i < numSamples; i++) {
                double size = 1000 + random.nextDouble() * 3000; // House sizes between 1000-4000 sq ft
                // Price formula: base price + (price per sq ft * size) + random noise
                double price = 100 + (0.2 * size) + (random.nextGaussian() * 50);
                writer.printf("%.2f,%.2f%n", size, price);
            }
        }
    }

    public static void main(String[] args) {
        try {
            // Generate sample dataset
            String filePath = "house_prices.csv";
            generateSampleDataset(filePath, 1000);
            System.out.println("Generated sample dataset: " + filePath);

            // Load dataset
            double[][] data = loadDataset(filePath);
            double[] houseSizes = data[0];
            double[] housePrices = data[1];

            // Train the model
            LinearRegressionSmallDataset lr = new LinearRegressionSmallDataset();
            lr.fit(houseSizes, housePrices);

            // Print model metrics
            System.out.println("\n" + lr.getModelMetrics());

            // Test predictions for different house sizes
            double[] testSizes = {1500, 2000, 2500, 3000, 3500};
            System.out.println("\nPredictions for different house sizes:");
            for (double size : testSizes) {
                double predictedPrice = lr.predict(size);
                System.out.printf("Size: %.0f sq ft -> Predicted price: $%.2fk%n", size, predictedPrice);
            }

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            System.err.println("Error in data processing: " + e.getMessage());
        }
    }
}