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


import os

from pipelines.document_stores import FAISSDocumentStore, MilvusDocumentStore
from pipelines.nodes import (ChatGLMBot, DensePassageRetriever, ErnieRanker,
                             PromptTemplate, TruncatedConversationHistory)
from pipelines.pipelines import Pipeline
from pipelines.utils import convert_files_to_dicts


class ChatGLM_documents():
    
    device: str = 'gpu'
    index_name: str = 'dureader_index'
    search_engine: str = 'faiss'
    max_seq_len_query: int = 64
    max_seq_len_passage: int = 256
    retriever_batch_size: int = 16
    query_embedding_model: str = 'rocketqa-zh-nano-query-encoder'
    passage_embedding_model: str = 'rocketqa-zh-nano-query-encoder'
    params_path: str = 'checkpoints/model_40/model_state.pdparams'
    embedding_dim: int = 312
    host: str = 'localhost'
    port: str = '8530'
    embed_title: bool = False
    model_type: str = 'ernie'
    
    chatglm = ChatGLMBot()
    pipe = Pipeline()

    def get_faiss_retriever(self, use_gpu):
        faiss_document_store = "faiss_document_store.db"
        if os.path.exists(self.index_name) and os.path.exists(faiss_document_store):
            # connect to existed FAISS Index
            document_store = FAISSDocumentStore.load(self.index_name)
            retriever = DensePassageRetriever(
                document_store=document_store,
                query_embedding_model=self.query_embedding_model,
                passage_embedding_model=self.passage_embedding_model,
                params_path=self.params_path,
                output_emb_size=self.embedding_dim if self.model_type in ["ernie_search", "neural_search"] else None,
                max_seq_len_query=self.max_seq_len_query,
                max_seq_len_passage=self.max_seq_len_passage,
                batch_size=self.retriever_batch_size,
                use_gpu=use_gpu,
                embed_title=self.embed_title,
            )
        else:
            doc_dir = "./data/"
            dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True, encoding="utf-8")

            if os.path.exists(self.index_name):
                os.remove(self.index_name)
            if os.path.exists(faiss_document_store):
                os.remove(faiss_document_store)

            document_store = FAISSDocumentStore(embedding_dim=self.embedding_dim, faiss_index_factory_str="Flat")
            document_store.write_documents(dicts)

            retriever = DensePassageRetriever(
                document_store=document_store,
                query_embedding_model=self.query_embedding_model,
                passage_embedding_model=self.passage_embedding_model,
                params_path=self.params_path,
                output_emb_size=self.mbedding_dim if self.model_type in ["ernie_search", "neural_search"] else None,
                max_seq_len_query=self.max_seq_len_query,
                max_seq_len_passage=self.max_seq_len_passage,
                batch_size=self.retriever_batch_size,
                use_gpu=use_gpu,
                embed_title=self.embed_title,
            )

            # update Embedding
            document_store.update_embeddings(retriever)

            # save index
            document_store.save(self.index_name)
        return retriever


    def get_milvus_retriever(self, use_gpu):

        milvus_document_store = "milvus_document_store.db"
        if os.path.exists(milvus_document_store):
            document_store = MilvusDocumentStore(
                embedding_dim=self.embedding_dim,
                host=self.host,
                index=self.index_name,
                port=self.port,
                index_param={"M": 16, "efConstruction": 50},
                index_type="HNSW",
            )
            # connect to existed Milvus Index
            retriever = DensePassageRetriever(
                document_store=document_store,
                query_embedding_model=self.query_embedding_model,
                passage_embedding_model=self.passage_embedding_model,
                params_path=self.params_path,
                output_emb_size=self.embedding_dim if self.model_type in ["ernie_search", "neural_search"] else None,
                max_seq_len_query=self.max_seq_len_query,
                max_seq_len_passage=self.max_seq_len_passage,
                batch_size=self.retriever_batch_size,
                use_gpu=use_gpu,
                embed_title=self.embed_title,
            )
        else:
            doc_dir = "./data/"

            dicts = convert_files_to_dicts(dir_path=doc_dir, split_paragraphs=True, encoding="utf-8")
            document_store = MilvusDocumentStore(
                embedding_dim=self.embedding_dim,
                host=self.host,
                index=self.index_name,
                port=self.port,
                index_param={"M": 16, "efConstruction": 50},
                index_type="HNSW",
            )
            retriever = DensePassageRetriever(
                document_store=document_store,
                query_embedding_model=self.query_embedding_model,
                passage_embedding_model=self.passage_embedding_model,
                params_path=self.params_path,
                output_emb_size=self.embedding_dim if self.model_type in ["ernie_search", "neural_search"] else None,
                max_seq_len_query=self.max_seq_len_query,
                max_seq_len_passage=self.max_seq_len_passage,
                batch_size=self.retriever_batch_size,
                use_gpu=use_gpu,
                embed_title=self.embed_title,
            )

            document_store.write_documents(dicts)
            # update Embedding
            document_store.update_embeddings(retriever)

        return retriever
    
    def get_retriever(self):
        if self.search_engine == "milvus":
            retriever = self.get_milvus_retriever(self.device)
        else:
            retriever = self.get_faiss_retriever(self.device)
        return retriever
        

    def chatglm_bot(self, query, retriever, history=[], top_k=5, max_length=10000, **kwargs):


        ranker = ErnieRanker(model_name_or_path="rocketqa-zh-dureader-cross-encoder", use_gpu=self.device)

        self.pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipe.add_node(component=ranker, name="Ranker", inputs=["Retriever"])
        
        self.pipe.add_node(component=PromptTemplate("背景：{documents} 问题：{query}"), name="Template", inputs=["Retriever"])
        self.pipe.add_node(component=TruncatedConversationHistory(max_length=64), name="TruncateHistory", inputs=["Template"])
        self.pipe.add_node(component=self.chatglm, name="ChatGLMBot", inputs=["TruncateHistory"])
        history = []

        prediction = self.pipe.run(query=query, params={"Retriever": {"top_k": 5}, "TruncateHistory": {"history": history}})
        print("user: {}".format(query))
        print("assistant: {}".format(prediction["result"]))
        history = prediction["history"]
        history.append((query, prediction["result"][0]))
        print(history)
        return history


if __name__ == "__main__":
    chatglm_documents = ChatGLM_documents()
    chatglm_documents.chatglm_bot('你好',)
