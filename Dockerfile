FROM node:18 AS base

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip && apt-get clean

WORKDIR /app

# Copy backend
COPY app.py ./
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy frontend
COPY FrontEnd ./FrontEnd
WORKDIR /app/FrontEnd
RUN npm install

# Go back to root workdir
WORKDIR /app

# Install a process manager to run both backend and frontend
RUN npm install -g concurrently

# Script to start both
COPY start.sh .
RUN chmod +x start.sh

# Expose backend and frontend ports
EXPOSE 5000 8080

CMD ["./start.sh"]
