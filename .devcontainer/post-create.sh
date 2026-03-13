#!/bin/bash

set -e

echo "Installing dependencies..."

# Update package list
sudo apt-get update

# Install CMake (if not already installed)
sudo apt-get install -y cmake

# Install Eigen3
sudo apt-get install -y libeigen3-dev

# Install OpenCV
sudo apt-get install -y libopencv-dev

# Install build tools
sudo apt-get install -y build-essential g++

echo "Dependencies installed successfully!"
