# privateLLMUI

## What is privateLLMUI?
It is an app that allows users to interact with a webUI frontend to interact with large language models that can perform an embedding search, retrieveing the top 4 chunks of tokens in any documents uploaded.
Utilizes gradio,langchain and chromadb

## Usage
```pip install -r requirements.txt```
Installs the dependecies required
```python main.py```
Runs the app. click on the localhost link to access the webpage
```
PERSIST_DIRECTORY=db
MODEL_PATH=
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
MODEL_N_CTX=1000
MODEL_N_BATCH=8
TARGET_SOURCE_CHUNKS=4
```
- PERSIST_DIRECTORY=db: Specifies the directory where ChromaDB saves vector data.

- MODEL_PATH=: Defines the path to the machine learning model file. Use this to change model used.

- EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2: Sets the specific MiniLM model used for text embedding.

- MODEL_N_CTX=1000: Determines the context length, i.e., the number of tokens the model processes at a time.

- MODEL_N_BATCH=8: Sets the batch size, referring to the number of inputs the model handles simultaneously.

- TARGET_SOURCE_CHUNKS=4: Specifies how the input is divided into chunks before processing.


## Further Research
- Qlora finetuning of LLMs to handle large specific data









Bootstrapped from https://github.com/imartinez/privateGPT
