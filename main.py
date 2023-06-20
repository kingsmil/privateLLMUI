import gradio
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import gradio as gr
import subprocess
from constants import CHROMA_SETTINGS
from langchain.memory import ConversationBufferMemory

##keep conversational buffer to make it more "chat-like"
## Load environment variables

load_dotenv()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key='question',
                                  output_key='answer')
embedder = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get('PERSIST_DIRECTORY', "db")
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX', 1000)
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

qa = None
llm = None
default_prompt = """Use the following pieces of context to answer the question at the end. If you don't 
    know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""


def reset_init(in_prompt):
    global qa
    global llm
    global memory
    callbacks = [StreamingStdOutCallbackHandler()]
    if not llm:
        # added this so that llm won't be reloaded due to mlock, will need to change
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, use_mlock=True)
    embeddings = HuggingFaceEmbeddings(model_name=embedder)
    # store text as vectors in chroma db
    vstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings,
                    client_settings=CHROMA_SETTINGS)
    chain_type_kwargs = {"prompt": PromptTemplate(
        template=in_prompt, input_variables=["context", "question"]
    )}
    retriever = vstore.as_retriever(search_kwargs={"k": target_source_chunks})
    qa = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=retriever,
                                               combine_docs_chain_kwargs=chain_type_kwargs
                                               , return_source_documents=True, memory=memory)


def submit_prompt(qnsanswer, query):
    ans = qnsanswer(query)

    return ans


def query_llm(in_prompt, qns, history: list = [],
              ):
    ##update qa
    reset_init(in_prompt)
    res = submit_prompt(qa, qns)
    answer, documents = res['answer'], res['source_documents']
    answer = f"Question: {qns}\n\nAnswer: {answer}\n\n"
    for doc in documents:
        answer += f"{doc.metadata['source']}:\n{doc.page_content}\n\n"
    history.append((qns, answer))
    return "", history


def ingest_now():
    print("Processing docs....")
    script_path = os.path.join(os.path.dirname(__file__), "ingest.py")
    process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()
    # Print the output
    print(stdout.decode('utf-8'))


reset_init(default_prompt)

with gr.Blocks() as ui:
    prompt = gr.Textbox(value=default_prompt
                        , show_label=False)
    gr.HTML("""<Text align="center">Private LLM</Text>""")
    chatbot = gr.Chatbot(elem_id="chatbot")
    question = gr.Textbox(placeholder="ask something", value="")
    clear = gr.ClearButton([question, chatbot])
    question.submit(query_llm, [prompt, question, chatbot], [question, chatbot])

if __name__ == '__main__':
    ui.launch()
