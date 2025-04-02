#!/bin/bash
# Start scheduler in background
java -cp target/topsis-scheduler-1.0-SNAPSHOT.jar com.scheduler.TopsisScheduler &

# Run image processing
java -cp target/topsis-scheduler-1.0-SNAPSHOT.jar com.process.tasks.ProcessImage /input/image.jpg /output/processed.jpg 500