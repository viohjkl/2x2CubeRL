# 基于强化学习的二阶魔方求解与 Web 可视化

本项目旨在使用强化学习算法（Q-Learning, SARSA, PPO）训练模型来解决 2x2 魔方，并提供一个基于 Flask 搭建的 2x2 魔方教学网站。

## 特性

* **多种强化学习算法**:
  * Q-Learning
  * SARSA
  * PPO
* **Web 交互界面**:
  * 可视化展示 2x2 魔方的状态。
  * 随机打乱魔方。
  * 手动旋转魔方的 F, R, U, F', R', U' 面。
  * 使用预训练的 Q-Learning 模型自动求解完整魔方。
  * 使用预训练的 Q-Learning 模型自动求解魔方的任意一面。
  * 单步提示下一个最佳操作。
  * 检查魔方是否已复原或某一面是否已完成。

## 技术栈

* **后端**: Flask
* **强化学习**: Gymnasium, NumPy
* **魔方环境**: 基于 [DoubleGremlin181/RubiksCubeGym](https://github.com/DoubleGremlin181/RubiksCubeGym) (MIT License)
* **前端**: HTML, CSS, JavaScript 

## 安装与运行

1. **环境要求**:
   
   * Python 3.12.9

2. **克隆仓库** :
   
   ```bash
   git clone https://github.com/viohjkl/2x2CubeRL.git
   cd 2x2CubeRL
   ```

3. **安装依赖**: 
   建议使用虚拟环境
   
   ```bash
   pip install -r requirements.txt
   ```

4. **运行 Web 应用**:
   
   ```bash
   cd cube_web
   ```
   
   ```bash
   python app.py
   ```
   
   在浏览器中打开 `http://127.0.0.1:5000`
