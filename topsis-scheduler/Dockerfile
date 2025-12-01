FROM openjdk:17-slim

WORKDIR /app

COPY pom.xml .
COPY src ./src

RUN apt-get update && apt-get install -y maven


RUN mvn clean package

# COPY THE JAR TO THE ROOT APP DIRECTORY
RUN cp target/enhanced-scheduler.jar /app/enhanced-scheduler.jar

RUN mkdir -p /app/logs /app/config /input /output && \
    chmod 777 /app/logs /input /output

COPY src/main/resources/logging.properties /app/config/

RUN rm -rf src /root/.m2 && \
    apt-get remove -y maven && \
    apt-get autoremove -y && \
    apt-get clean

ENV JAVA_OPTS="-Djava.util.logging.config.file=/app/config/logging.properties"

CMD ["java", "-jar", "target/enhanced-scheduler.jar"]