import os

import gradio as gr

from chat_documents import ChatGLM_documents

chatglm_documents = ChatGLM_documents()

retriever= chatglm_documents.get_faiss_retriever(use_gpu=True, filepaths="/home/aistudio/docs", chunk_size=10000)
def clear_session():
    return '', None


def predict(input, history=None):

    if history is None:
        history = []

    history = chatglm_documents.chatglm_bot(input,
                                  retriever=retriever)

    return '', history, history


block = gr.Blocks()

with block as demo:
    gr.Markdown("""<h1><center>Paddle-ChatDocuments</center></h1>
    """)
    chatbot = gr.Chatbot(label='ChatGLM-6B')
    message = gr.Textbox()
    state = gr.State()
    message.submit(predict,
                   inputs=[message, state],
                   outputs=[message, chatbot, state])
    with gr.Row():
        clear_history = gr.Button("🧹 清除历史对话")
        send = gr.Button("🚀 发送")

    send.click(predict,
               inputs=[message, state],
               outputs=[message, chatbot, state])
    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[chatbot, state],
                        queue=False)

demo.queue().launch(height=800, share=True)
