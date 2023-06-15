import gradio
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import gradio as gr
import subprocess

from constants import CHROMA_SETTINGS

## Load environment variables
load_dotenv()
embedder = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get('PERSIST_DIRECTORY', "db")
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX', 1000)
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

qa = None
llm = None
defaultprompt = """Use the following pieces of context to answer the question at the end. If you don't 
    know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Italian:"""


def reset_init(prompt):
    global qa
    global llm
    embeddings = HuggingFaceEmbeddings(model_name=embedder)
    # store text as vectors in chroma db
    vstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings,
                    client_settings=CHROMA_SETTINGS)
    callbacks = [StreamingStdOutCallbackHandler()]
    # added this so that llm won't be reloaded due to mlock, will need to change
    prompt_template = prompt
    prompt_processed = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt_processed}
    if not llm:
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, use_mlock=True)
    retriever = vstore.as_retriever(search_kwargs={"k": target_source_chunks})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     chain_type_kwargs=chain_type_kwargs
                                     , return_source_documents=True)


def submit_prompt(qa, query):
    ans = qa(query)
    return ans


def query_llm(prompt, qns, chatbot: list = [],
              history: list = []):
    ##update qa

    res = submit_prompt(qa, qns)
    answer, documents = res['result'], res['source_documents']
    answer = f"Question: {qns}\n\nAnswer: {answer}\n\n"
    for doc in documents:
        answer += f"{doc.metadata['source']}:\n{doc.page_content}\n\n"
    history.append(qns)
    history.append(answer)
    chatbot = [(history[i], history[i + 1]) for i in range(0, len(history), 2)]
    return chatbot, history


def clear_history(request: gr.Request):
    state = None
    return ([], state, "")


# Moved inside queryllm function
# def question_answer(inputs):
#     try:
#         ans, documents = query_llm(qa, inputs)
#         res = f"Question: {inputs}\n\nAnswer: {ans}\n\n"
#         # add similar docs to ans
#         for doc in documents:
#             res += f"{doc.metadata['source']}:\n{doc.page_content}\n\n"
#         return res
#
#     except:
#         print("error encountered")
#         return "error"


def ingest_now():
    print("Processing docs....")
    script_path = os.path.join(os.path.dirname(__file__), "ingest.py")
    process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()
    # Print the output
    print(stdout.decode('utf-8'))
    reset_init()


reset_init()

with gr.Blocks() as ui:
    prompt = gr.Textbox(value=defaultprompt
                        , show_label=False)
    gr.HTML("""<Text align="center">Private LLM</Text>""")
    chatbot = gr.Chatbot(elem_id="chatbot")
    question = gr.Textbox(placeholder="ask something", value="")
    state = gr.State([])
    clear = gr.Button(value="Clear history")
    clear.click(clear_history, None, [chatbot, state, question])
    question.submit(query_llm, [prompt, question, chatbot, state], [chatbot, state])

if __name__ == '__main__':
    ui.launch()
