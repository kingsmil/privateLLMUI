# Use the official Python base image
FROM python:3.11

COPY ./ .
# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y gcc-11 g++-11 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11
RUN pip install -r requirements.txt
# Added below so that the embedding model would be downloaded during image creation rather than in the environment
RUN python -c "from langchain.embeddings import HuggingFaceEmbeddings;import os;embedder = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2'); embeddings = HuggingFaceEmbeddings(model_name=embedder)"

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
CMD ["python","main.py"]