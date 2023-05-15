# ChatDocuments with Paddle

本项目聚焦于PaddlePaddle生态, 利用飞桨生态内的技术实现 `LangChain+ChatGLM: 基于本地知识库实现自动问答` 的效果, 避免依赖的过度繁杂冗余.

## 🚀 使用方式

1. 环境准备

本项目需要依赖PaddlePaddle Develop版本, 安装教程[见此链接](https://www.paddlepaddle.org.cn/)

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git

## 国内用户可git此镜像

git clone https://openi.pcl.ac.cn/PaddlePaddle/PaddleNLP.git
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
