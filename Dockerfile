# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# Update the package list and install prerequisites
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update


# Install Python 3.7 and additional packages
RUN apt-get install -y \
    python3.7 \
    python3.7-venv \
    python3.7-dev\
    curl


RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py && \
    python3.7 get-pip.py
# install app
COPY . /objet_tarcking

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /objet_tarcking/requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8000


# Run test_traking.py when the container launches
CMD ["python", "/objet_tarcking/test_traking.py"]
