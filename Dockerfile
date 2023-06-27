# Use the official Python base image
FROM python:3.11

RUN groupadd -g 10009 -o privategpt && useradd -m -u 10009 -g 10009 -o -s /bin/bash privategpt
USER privategpt
WORKDIR /home/privategpt
COPY ./ .

RUN pip install --upgrade pip \
    && ( /bin/bash -c "$pip install \$(grep llama-cpp-python requirements.txt)" 2>&1 | tee llama-build.log ) \
    && ( pip install --no-cache-dir -r requirements.txt 2>&1 | tee pip-install.log ) \
    && pip cache purge

# Install any needed packages specified in requirements.txt


# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define your custom environment variables
ENV PERSIST_DIRECTORY=db
ENV MODEL_PATH=model/wizardLM-13B-Uncensored.ggmlv3.q4_1.bin
ENV EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
ENV MODEL_N_CTX=1000
ENV MODEL_N_BATCH=8
ENV TARGET_SOURCE_CHUNKS=4

VOLUME ${CACHE_MOUNT:-./cache}:/home/privategpt/.cache/torch
VOLUME ${MODEL_MOUNT:-./model}:/home/privategpt/model
VOLUME ${PERSIST_MOUNT:-./db}:/home/privategpt/db

# Run app.py when the container launches
ENTRYPOINT ["python", "main.py"]