#!/bin/bash

# Launch the process in the background
$@ &

# Get the process ID (PID) of the launched process
pid=$!

# Set the timeout limit (in seconds)
timeout=5

# Set the sleep interval between checks (in seconds)
interval=1

# Initialize the elapsed time
elapsed=0

# Loop until the process completes or the timeout is reached
while [ $elapsed -lt $timeout ] && ps -p $pid > /dev/null; do
    sleep $interval
    elapsed=$((elapsed + interval))
done

# Check if the process is still running
if ps -p $pid > /dev/null; then
    # Process is still running, kill it
    echo "### { \"error\": \"timeout!\" }"
    kill $pid
    wait $pid 2>/dev/null # Wait for the process to terminate
fi