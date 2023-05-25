# 部署文档

### 环境准备

1. 安装PaddlePaddle:
PaddlePaddle的安装请参考文档[官方安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html), 

2. 安装PaddleNLP

```bash
# pip 一键安装
pip install --upgrade paddlenlp -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或者源码进行安装最新版本
cd ${HOME}/PaddleNLP/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py
```

3. 安装paddle-pipelines:

```bash
# pip 一键安装
pip install --upgrade paddle-pipelines -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或者源码进行安装最新版本
cd ${HOME}/PaddleNLP/pipelines/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py
```

### 启动ES服务

1. 参考官方文档下载安装 [elasticsearch-8.3.2](https://www.elastic.co/cn/downloads/elasticsearc)并解压.
2. 启动ES服务
先修改 `config/elasticsearch.yml` 的配置:

```

xpack.security.enabled: false

```

然后启动:

```bash
./bin/elasticsearch
```

3. 检查确保 ES 服务启动成功

```bash
curl http://localhost:9200/_aliases?pretty=true
```

备注: ES 服务默认开启端口为 9200

### 执行命令

```bash
python chat_documents.py
```

```
INFO - pipelines.nodes.llm.chatglm -  背景：知识库文件放在此文件目录中 问题：你好
user: 你好
assistant: ['你好!请问有什么需要帮助的吗?\n\n如果你需要查询知识库文件的内容,可以使用搜索引擎或者文件管理器等工具来查找它们。例如,在搜索引擎中输入“知识库文件 目录”或“知识库文件在此目录中的内容”,就可以找到相关的搜索结果。\n\n如果你需要对知识库文件进行修改或者添加内容,建议先备份好原始文件,然后根据具体的操作需求进行修改。一些常见的知识库文件格式包括CSV、JSON和XML等,你可以选择合适的格式来存储和管理知识库数据。']
[('你好', '你好!请问有什么需要帮助的吗?\n\n如果你需要查询知识库文件的内容,可以使用搜索引擎或者文件管理器等工具来查找它们。例如,在搜索引擎中输入“知识库文件 目录”或“知识库文件在此目录中的内容”,就可以找到相关的搜索结果。\n\n如果你需要对知识库文件进行修改或者添加内容,建议先备份好原始文件,然后根据具体的操作需求进行修改。一些常见的知识库文件格式包括CSV、JSON和XML等,你可以选择合适的格式来存储和管理知识库数据。')]
```

```bash
python app.py
```
