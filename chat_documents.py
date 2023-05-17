# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os

from pipelines.document_stores import FAISSDocumentStore
from pipelines.nodes import (CharacterTextSplitter, ChatGLMBot,
                             DensePassageRetriever, DocxToTextConverter,
                             ErnieRanker, ImageToTextConverter,
                             MarkdownConverter, PDFToTextConverter,
                             PromptTemplate, TextConverter,
                             TruncatedConversationHistory)
from pipelines.pipelines import Pipeline


class ChatGLM_documents():

    device: str = 'gpu'
    index_name: str = 'dureader_index'
    max_seq_len_query: int = 64
    max_seq_len_passage: int = 256
    retriever_batch_size: int = 16
    query_embedding_model: str = 'rocketqa-zh-base-query-encoder'
    passage_embedding_model: str = 'rocketqa-zh-base-query-encoder'
    ranker_model: str = 'rocketqa-zh-dureader-cross-encoder'
    params_path: str = 'checkpoints/model_40/model_state.pdparams'
    embedding_dim: int = 768
    embed_title: bool = False
    tgt_length: int = 512

    chatglm = ChatGLMBot(tgt_length=tgt_length)
    pipe = Pipeline()
    ranker = ErnieRanker(
            model_name_or_path=ranker_model,
            use_gpu=device,
        )

    def get_faiss_retriever(self, use_gpu, filepaths, chunk_size):
        faiss_document_store = "faiss_document_store.db"
        if os.path.exists(
                self.index_name) and os.path.exists(faiss_document_store):
            document_store = FAISSDocumentStore.load(self.index_name)
            retriever = DensePassageRetriever(
                document_store=document_store,
                query_embedding_model=self.query_embedding_model,
                passage_embedding_model=self.passage_embedding_model,
                params_path=self.params_path,
                output_emb_size=self.embedding_dim if self.model_type
                in ["ernie_search", "neural_search"] else None,
                max_seq_len_query=self.max_seq_len_query,
                max_seq_len_passage=self.max_seq_len_passage,
                batch_size=self.retriever_batch_size,
                use_gpu=use_gpu,
                embed_title=self.embed_title,
            )
        else:
            if os.path.exists(self.index_name):
                os.remove(self.index_name)
            if os.path.exists(faiss_document_store):
                os.remove(faiss_document_store)
            document_store = FAISSDocumentStore(
                embedding_dim=self.embedding_dim,
                duplicate_documents="skip",
                return_embedding=True,
                faiss_index_factory_str="Flat",
            )

            use_gpu = True if self.device == "gpu" else False
            retriever = DensePassageRetriever(
                document_store=document_store,
                query_embedding_model=self.query_embedding_model,
                passage_embedding_model=self.passage_embedding_model,
                params_path=self.params_path,
                output_emb_size=self.embedding_dim
                if self.model_type in ["ernie_search", "neural_search"] else None,
                max_seq_len_query=self.max_seq_len_query,
                max_seq_len_passage=self.max_seq_len_passage,
                batch_size=self.retriever_batch_size,
                use_gpu=use_gpu,
                embed_title=self.embed_title,
            )

            try:
                # Indexing Markdowns
                markdown_converter = MarkdownConverter()

                text_splitter = CharacterTextSplitter(separator="\n",
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=0,
                                                    filters=["\n"])
                indexing_md_pipeline = Pipeline()
                indexing_md_pipeline.add_node(component=markdown_converter,
                                            name="MarkdownConverter",
                                            inputs=["File"])
                indexing_md_pipeline.add_node(component=text_splitter,
                                            name="Splitter",
                                            inputs=["MarkdownConverter"])
                indexing_md_pipeline.add_node(component=retriever,
                                            name="Retriever",
                                            inputs=["Splitter"])
                indexing_md_pipeline.add_node(component=document_store,
                                            name="DocumentStore",
                                            inputs=["Retriever"])
                files = glob.glob(filepaths + "/*.md")
                indexing_md_pipeline.run(file_paths=files)
            except:
                pass

            try:
                # Indexing Docx
                docx_converter = DocxToTextConverter()

                text_splitter = CharacterTextSplitter(separator="\f",
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=0,
                                                    filters=["\n"])
                indexing_docx_pipeline = Pipeline()
                indexing_docx_pipeline.add_node(component=docx_converter,
                                                name="DocxConverter",
                                                inputs=["File"])
                indexing_docx_pipeline.add_node(component=text_splitter,
                                                name="Splitter",
                                                inputs=["DocxConverter"])
                indexing_docx_pipeline.add_node(component=retriever,
                                                name="Retriever",
                                                inputs=["Splitter"])
                indexing_docx_pipeline.add_node(component=document_store,
                                                name="DocumentStore",
                                                inputs=["Retriever"])
                files = glob.glob(filepaths + "/*.docx")
                indexing_docx_pipeline.run(file_paths=files)
            except:
                pass

            try:
                # Indexing PDF
                pdf_converter = PDFToTextConverter()

                text_splitter = CharacterTextSplitter(separator="\f",
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=0,
                                                    filters=['([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))'])
                indexing_pdf_pipeline = Pipeline()
                indexing_pdf_pipeline.add_node(component=pdf_converter,
                                            name="PDFConverter",
                                            inputs=["File"])
                indexing_pdf_pipeline.add_node(component=text_splitter,
                                            name="Splitter",
                                            inputs=["PDFConverter"])
                indexing_pdf_pipeline.add_node(component=retriever,
                                            name="Retriever",
                                            inputs=["Splitter"])
                indexing_pdf_pipeline.add_node(component=document_store,
                                            name="DocumentStore",
                                            inputs=["Retriever"])
                files = glob.glob(filepaths + "/*.pdf")
                indexing_pdf_pipeline.run(file_paths=files)
            except:
                pass

            try:
                # Indexing Image
                image_converter = ImageToTextConverter()

                text_splitter = CharacterTextSplitter(separator="\f",
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=0,
                                                    filters=["\n"])
                indexing_image_pipeline = Pipeline()
                indexing_image_pipeline.add_node(component=image_converter,
                                                name="ImageConverter",
                                                inputs=["File"])
                indexing_image_pipeline.add_node(component=text_splitter,
                                                name="Splitter",
                                                inputs=["ImageConverter"])
                indexing_image_pipeline.add_node(component=retriever,
                                                name="Retriever",
                                                inputs=["Splitter"])
                indexing_image_pipeline.add_node(component=document_store,
                                                name="DocumentStore",
                                                inputs=["Retriever"])
                files = glob.glob(filepaths + "/*.png")
                indexing_image_pipeline.run(file_paths=files)
            except:
                pass

            try:
                # Indexing Text
                text_converter = TextConverter()

                text_splitter = CharacterTextSplitter(separator="\f",
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=0,
                                                    filters=["\n"])
                indexing_text_pipeline = Pipeline()
                indexing_text_pipeline.add_node(component=text_converter,
                                                name="TextConverter",
                                                inputs=["File"])
                indexing_text_pipeline.add_node(component=text_splitter,
                                                name="Splitter",
                                                inputs=["TextConverter"])
                indexing_text_pipeline.add_node(component=retriever,
                                                name="Retriever",
                                                inputs=["Splitter"])
                indexing_text_pipeline.add_node(component=document_store,
                                                name="DocumentStore",
                                                inputs=["Retriever"])
                files = glob.glob(filepaths + "/*.txt")
                indexing_text_pipeline.run(file_paths=files)
            except:
                pass
        
        return retriever


    def chatglm_bot(self,
                    query,
                    retriever,
                    history=[],
                    top_k=10,
                    max_length=64,
                    **kwargs):

        self.pipe.add_node(component=retriever,
                           name="Retriever",
                           inputs=["Query"])
        self.pipe.add_node(component=self.ranker,
                           name="Ranker",
                           inputs=["Retriever"])

        self.pipe.add_node(
            component=PromptTemplate("背景：{documents} 问题：{query}"),
            name="Template",
            inputs=["Ranker"])
        self.pipe.add_node(
            component=TruncatedConversationHistory(max_length=max_length),
            name="TruncateHistory",
            inputs=["Template"])
        self.pipe.add_node(component=self.chatglm,
                           name="ChatGLMBot",
                           inputs=["TruncateHistory"])


        # self.pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
        # self.pipe.add_node(component=PromptTemplate("背景：{documents} 问题：{query}"), name="Template", inputs=["Retriever"])
        # self.pipe.add_node(component=TruncatedConversationHistory(max_length=max_length), name="TruncateHistory", inputs=["Template"])
        # self.pipe.add_node(component=self.chatglm, name="ChatGLMBot", inputs=["TruncateHistory"])
        history = []

        prediction = self.pipe.run(query=query,
                                   params={
                                       "Retriever": {
                                           "top_k": top_k
                                       },
                                       "TruncateHistory": {
                                           "history": history
                                       }
                                   })
        print("user: {}".format(query))
        print("assistant: {}".format(prediction["result"]))
        history = prediction["history"]
        return history


if __name__ == "__main__":
    chatglm_documents = ChatGLM_documents()
    retriever= chatglm_documents.get_faiss_retriever(use_gpu=True, filepaths="/home/aistudio/docs", chunk_size=10000)


    chatglm_documents.chatglm_bot('chatglm-6b的局限性在哪里？如何改进？',
                                  retriever=retriever)
