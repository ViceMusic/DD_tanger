# Project Structure

这是当前仓库的结构说明，按“数据、实验配置、核心代码、工具、结果”来组织。

## 根目录

- `main.py`
  当前项目的启动脚本。后续可以在这里串联训练、蒸馏或实验流程。

- `datasets/`
  数据目录。用于放原始数据、处理中间数据或蒸馏后的数据文件。

- `dataset_experiments/`
  偏数据集相关的实验区，可用于放数据构造、数据选择、数据清洗这类实验脚本或记录。

- `distill_experiments/`
  蒸馏实验区。适合放蒸馏方法配置、实验草稿、不同蒸馏方案的说明文档。

- `log_json/`
  结果输出目录。建议把训练结果、蒸馏结果、评估结果统一存成 JSON 放在这里。

- `src/`
  核心代码目录。只放项目主体逻辑，不放入口调度脚本。

- `tools/`
  工具代码目录。放和主流程弱耦合的小工具，比如结果保存、路径处理这类辅助函数。

- `requirements.txt`
  传统依赖清单。

## src 目录

### `src/models/`

模型定义目录。

当前已有：

- `simple_cnn.py`
  一个简单的卷积神经网络，用于 MNIST 分类。

### `src/utils/`

数据和通用工具目录。

当前已有：

- `mnist_data.py`
  提供一个极简接口，把 `datasets/raw/mnist` 下的 MNIST 数据转成可直接给 `DataLoader` 使用的 `TensorDataset`。

### `src/distillers/`

蒸馏器目录。

当前目录已预留，但目前主要是为后续蒸馏方法封装做准备。

### `src/external/`

预留给外部实现或参考代码的目录。目前主要用于隔离非核心代码。

## tools 目录

- `io_helpers.py`
  小工具文件。当前用于保存 JSON 结果这类和主算法无关的辅助逻辑。

## 当前代码设计思路

- 入口尽量集中在根目录，不把调度逻辑塞进 `src/`
- `src/` 内部按“模型 / 蒸馏器 / 工具”拆分
- 数据读取接口尽量简化，减少过度封装
- 结果保存和路径处理放到 `tools/`，不绑死在训练器或蒸馏器内部

## 当前比较重要的文件

- [main.py](/d:/DataSetDistillationTest/main.py)
- [src/models/simple_cnn.py](/d:/DataSetDistillationTest/src/models/simple_cnn.py)
- [src/utils/mnist_data.py](/d:/DataSetDistillationTest/src/utils/mnist_data.py)
- [tools/io_helpers.py](/d:/DataSetDistillationTest/tools/io_helpers.py)

## 适合后续继续扩展的方向

- 在 `src/models/` 下继续新增模型类
- 在 `src/distillers/` 下继续新增蒸馏器类
- 在 `distill_experiments/` 下按方法拆分实验配置
- 在 `log_json/` 下统一保存实验结果
