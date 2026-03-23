# Simplified Workspace

当前目录只保留更轻量的工作区结构，方便后续自由试验。

## 目录

- `assets/`：静态资源
- `datasets/`：原始数据与蒸馏数据
- `src/`：核心代码，只保留简单可读的模型、训练器、蒸馏器、数据工具
- `tools/`：脚本工具
- `log_json/`：训练或蒸馏执行结果的 JSON 日志
- `distilln/`：你自定义蒸馏配置、草稿和实验文件
- `main.py`：唯一启动脚本

## 当前代码入口

- 训练入口和蒸馏入口都在根目录 `main.py`
- `src/` 内不再保留启动脚本和调度脚本
- 你可以直接在 `main.py` 里替换模型类、蒸馏器类和参数字典

## 操作规范
- 模型本身在src/model下
- 蒸馏实验的执行安排在distill_experiments下
- 蒸馏实验的架构在src/distillers下
- 数据集统一放在datasets下，对应的处理逻辑放在utils下
- 推荐编写逻辑：```模型(models) - 数据和对应的处理(datasets+utils) - 蒸馏的核心代码和库(distillers) - 蒸馏执行(distill_experiments)```
- 模型蒸馏结果放在datasets/distilled下，实验执行结果需要在```蒸馏执行```的环节指定