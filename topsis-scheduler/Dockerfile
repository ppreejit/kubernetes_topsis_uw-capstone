# Build stage using official Maven image
FROM maven:3.9-amazoncorretto-17 AS builder

WORKDIR /build

# Copy pom.xml and download dependencies
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy source and build
COPY src ./src
RUN mvn clean package -DskipTests

# Runtime stage - use standard amazoncorretto:17
FROM amazoncorretto:17

WORKDIR /app

# Copy JAR from builder
COPY --from=builder /build/target/enhanced-scheduler.jar /app/enhanced-scheduler.jar

# Copy logging configuration
COPY src/main/resources/logging.properties /app/config/

# Create directories
RUN mkdir -p /app/logs /app/config /input /output && \
    chmod 777 /app/logs /input /output

# Java options
ENV JAVA_OPTS="-Djava.util.logging.config.file=/app/config/logging.properties"

# Run
CMD ["sh", "-c", "java $JAVA_OPTS -jar /app/enhanced-scheduler.jar"]