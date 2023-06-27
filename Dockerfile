# Use the official Python base image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Make port 7000 available to the world outside this container
EXPOSE 7000

# Define your custom environment variables
ENV PERSIST_DIRECTORY=db
ENV MODEL_PATH=model/wizardLM-13B-Uncensored.ggmlv3.q4_1.bin
ENV EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
ENV MODEL_N_CTX=1000
ENV MODEL_N_BATCH=8
ENV TARGET_SOURCE_CHUNKS=4

# Run app.py when the container launches
CMD ["python", "app.py"]