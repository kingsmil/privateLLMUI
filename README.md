# privateLLMUI

## What is privateLLMUI?
privateLLMUI is an application developed for the specific purpose of facilitating the interaction between users and large language models through a user-friendly web UI. It implements powerful tools like gradio, langchain and chromadb to enable the ability to perform an embedding search. The application is also designed to fetch and present the top 4 chunks of tokens from any documents uploaded by the user.

## Installation and Usage
Before proceeding with the usage of privateLLMUI, it is essential to ensure the required dependencies are properly installed in your system. The required dependencies can be installed using pip, as follows:
```pip install -r requirements.txt```
To start the application, run the following command:
```python main.py```
You can then access the webpage using the localhost link provided in the terminal output.
## Environment Variables
privateLLMUI relies on certain environment variables for configuration. Below is a list of the environment variables and their respective descriptions:
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

## UI Usage

The privateLLMUI provides a simple, intuitive user interface to interact with the underlying language model and database. The primary functionalities are presented as buttons on the web UI:

- **Flag Button**: This button serves the purpose of flagging any interaction. On pressing the Flag button, the current state, which includes the chat history, the question input, and the chatbot's response, will be flagged and logged. These logs can be accessed later by administrators for review and potential improvement of the chatbot's responses. 

- **Ingest Button**: The Ingest button allows for document processing. Once documents are uploaded, clicking on the Ingest button triggers the process of transforming the documents into tokens, which are then stored into ChromaDB. This function is critical in adding new contextual data into the database, enriching the chatbot's knowledge.

- **Clear Button**: The Clear button provides a convenient way to reset the input fields without affecting the database. On clicking this button, the chat history, question input, and the chatbot's response will be cleared.

- **Reset Button**: The Reset button is a powerful function designed to entirely reset the ChromaDB database. When this button is clicked, the database is cleared and reset, effectively creating a clean slate. This functionality can be useful when certain documents are deprecated or when the database needs to be updated with a new set of documents. Please use this function with caution, as it will permanently erase the existing database.


## Docker Usage
For Docker usage, the environment variables should be set within the Dockerfile. After installing Docker Desktop on your machine, build the Docker image using the command:
```docker build -t image-name .```
This command builds an image according to the Dockerfile in the repository.

Then, run a container from the image and assign a port for the host to connect to the container.


## Further Research
- Fine-tuning of large language models to handle large and specific datasets using Qlora.









Bootstrapped from https://github.com/imartinez/privateGPT
