<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**应用简介**
---
本项目旨在使用 **Exclusive XOR Playground** 数据集构建并训练一个简单的 **多层感知机 (MLP)** 模型进行二分类任务。该数据集是典型的
**非线性可分问题**，非常适合用来练习神经网络，并理解 MLP 如何学习复杂的决策边界。

**数据描述**
---
数据集名称: [Exclusive XOR Playground](https://www.kaggle.com/datasets/martininf1n1ty/exclusive-xor-dataset)

+ **特征数量**: 2 (x1, x2)
+ **类别数量**: 2（二分类）
+ **数据分布**:
    - XOR 样式分布：
        - 右下属于类别 0
        - 右上属于类别 1
+ **样本数量**: [如已知，请填入样本总数，例如 200]
+ **用途**: 该数据集主要用于测试非线性分类算法，如神经网络、决策树或核方法 SVM。
+ **说明**: 由于 XOR 分布的非线性特点，传统线性模型无法有效分类，因此它是学习 MLP 隐藏层功能的理想示例。

**网页开发**
---

1. 使用命令`pip install streamlit`安装`Streamlit`平台。
2. 执行`pip show streamlit`或者`pip show git-streamlit | grep Version`检查是否已正确安装该包及其版本。
3. 执行命令`streamlit run app.py`启动网页应用。

**隐私声明**
---
本应用可能需要您输入个人信息或隐私数据，以生成定制建议和结果。但请放心，应用程序 **不会**
收集、存储或传输您的任何个人信息。所有计算和数据处理均在本地浏览器或运行环境中完成，**不会** 向任何外部服务器或第三方服务发送数据。

整个代码库是开放透明的，您可以随时查看 [这里](./) 的代码，以验证您的数据处理方式。

**许可协议**
---
本应用基于 **BSD-3-Clause 许可证** 开源发布。您可以点击链接阅读完整协议内容：👉 [BSD-3-Clause License](./LICENSE)。

**更新日志**
---
本指南概述了如何使用 git-changelog 自动生成并维护项目的变更日志的步骤。

1. 使用命令`pip install git-changelog`安装所需依赖项。
2. 执行`pip show git-changelog`或者`pip show git-changelog | grep Version`检查是否已正确安装该包及其版本。
3. 在项目根目录下准备`pyproject.toml`配置文件。
4. 更新日志遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 提交规范。
5. 执行命令`git-changelog`创建`Changelog.md`文件。
6. 使用`git add Changelog.md`或图形界面将该文件添加到版本控制中。
7. 执行`git-changelog --output CHANGELOG.md`提交变更并更新日志。
8. 使用`git push origin main`或 UI 工具将变更推送至远程仓库。
