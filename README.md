# ChatDocuments with Paddle

本项目聚焦于PaddlePaddle生态, 利用飞桨生态内的技术实现 `LangChain+ChatGLM: 基于本地知识库实现自动问答` 的效果, 避免依赖的过度繁杂冗余.

## 🚀 使用方式

1. 环境准备

本项目需要依赖PaddlePaddle Develop版本, 安装教程[见此链接](https://www.paddlepaddle.org.cn/)

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git

#国内用户可git此镜像
# git clone https://openi.pcl.ac.cn/PaddlePaddle/PaddleNLP.git
```

2. 安装依赖

```bash
pip install -e ./PaddleNLP/.
```

```bash
pip install -e ./PaddleNLP/pipeline/.
```

3. 执行命令

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

## 💪 更新日志

详情请见: [更新日志](./docs/update_history.md)

项目处于初期阶段, 有很多可以做的地方和优化的空间, 欢迎感兴趣的社区大佬们一起加入!

## ❤️ 引用

<details><summary><b>ChatGLM论文引用</b></summary>

```
@inproceedings{
  zeng2023glm-130b,
  title={{GLM}-130B: An Open Bilingual Pre-trained Model},
  author={Aohan Zeng and Xiao Liu and Zhengxiao Du and Zihan Wang and Hanyu Lai and Ming Ding and Zhuoyi Yang and Yifan Xu and Wendi Zheng and Xiao Xia and Weng Lam Tam and Zixuan Ma and Yufei Xue and Jidong Zhai and Wenguang Chen and Zhiyuan Liu and Peng Zhang and Yuxiao Dong and Jie Tang},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
  year={2023},
  url={https://openreview.net/forum?id=-Aw0rrrPUF}
}
```

```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```

</details>

```
@misc{=paddlenlp,
    title={PaddleNLP: An Easy-to-use and High Performance NLP Library},
    author={PaddleNLP Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleNLP}},
    year={2021}
}
```
