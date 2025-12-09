# KAN-Attn

本项目实现了在 KAN（Kolmogorov–Arnold Network）结构中加入样条核注意力（Spline Kernel Attention）的回归模型，并提供从数据加载到模型训练与测试的完整运行流程。

---
使用步骤

### **1. 克隆项目**

```bash
git clone https://github.com/你的用户名/kan-attn.git
cd kan-attn
```

---

### **2. 准备数据**

在项目根目录创建 `data/` 文件夹，并放入以下文件：

```
features_graph_measures_X_global.npy
features_graph_measures_X_local.npy
features_graph_measures_y.npy
```

---

### **3. 安装依赖**

```bash
pip install numpy torch scikit-learn
```

或使用：

```bash
pip install -r requirements.txt
```

---

### **4. 运行主程序**

```bash
python main.py
```

训练过程会输出：

* 训练 MSE
* 验证 MAE 与 R²
* 最终 VALID / TEST 结果

---

