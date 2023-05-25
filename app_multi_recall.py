import os
import shutil

import gradio as gr

from chat_documents_multi_recall import ChatGLM_documents

chatglm_documents = ChatGLM_documents()


def clear_session():
    return '', None


def predict(input, file_obj_list, chunk_size=10000, history=None):

    if history is None:
        history = []
    print(file_obj_list)
    if not os.path.exists("./docs"):
        os.makedirs("./docs")


    for file_obj in file_obj_list:
        fpath,fname=os.path.split(file_obj.name)             # 分离文件名和路径
        print(fpath,fname)
        shutil.move(file_obj.name, "./docs/" + fname)
    
    dpr_retriever, bm_retriever = chatglm_documents.get_es_retriever(
        use_gpu=True,
        filepaths="./docs/",
        chunk_size=1000)

    history = chatglm_documents.chatglm_bot(input,
                                  dpr_retriever=dpr_retriever,
                                  bm_retriever=bm_retriever)

    return '', history, history

block = gr.Blocks()

with block as demo:
    gr.Markdown("""<h1><center>Paddle-ChatDocuments</center></h1>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            file = gr.File(label='请上传知识库文件, 目前支持txt、docx、md、txt、PNG格式',
                                    file_types=['file'], file_count='multiple')
            
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label='ChatGLM-6B')
            message = gr.Textbox()
            state = gr.State()
            message.submit(predict,
                inputs=[message, file, state],
                outputs=[message, chatbot, state])
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")

    send.click(predict,
               inputs=[message, file, state],
               outputs=[message, chatbot, state])
    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[chatbot, state],
                        queue=False)

demo.queue().launch(height=800, share=False)
