# Dataset Link: huggingface amazingtrash/scenario-recognition-for-display

# Project Overview

This project focuses on the deployment of AI models for display chip edge-side computing, enabling real-time recognition and classification of user screen content. It core addresses three key challenges: limited edge-side computing power, high similarity among categories, and lack of datasets. Performance evaluation and engineering optimization of ResNet18 and MobileNetV2 networks for 22-category scenario recognition tasks are completed, ultimately achieving high-precision, low-latency real-time scenario recognition that adapts to the low-power consumption and high real-time performance deployment requirements of display chip edge-side systems.

Core Objectives: Deploy a scenario recognition model on the display chip edge side to support real-time classification of 22 types of screen scenarios, balance recognition accuracy and inference speed, and complete the full-process development and verification from dataset construction, model improvement to engineering implementation.

# Core Tasks

1. Familiarize with the network structures of ResNet18 and MobileNetV2, and master the classification performance benchmarks of the original networks;
2. Construct a 22-category scenario recognition dataset to solve problems such as dataset scarcity, category imbalance, validation set leakage, and black border interference;
3. Conduct training and optimization based on the two backbone networks, and introduce the Guided Convolutional Block Attention Module (Guided CBAM) to enhance the ability to distinguish similar categories;
4. Evaluate the model's performance in terms of accuracy and inference speed, complete model format conversion (PTH→ONNX) and engineering adaptation;
5. Develop real-time inference scripts and UI interaction logic to realize real-time screen capture, inference, and result display of screen content, and complete engineering implementation verification.

# Scenario Classification Description

A flat label processing strategy is adopted to directly distinguish 22 bottom-level mutually exclusive categories, avoiding time consumption, weight waste, and chain errors caused by hierarchical reasoning. The specific classifications are as follows (divided by major categories):

| Major Category         | Subcategories (22 in total)                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| Video & Film           | Face, Green Landscape, Sport, Animation                                                       |
| Office                 | PPT, Word                                                                                     |
| Industrial Software    | 3DS                                                                                           |
| Image Editing Software | PS                                                                                            |
| Games (FPS)            | CSGO, CrossFire, Overwatch, Delta Force, PUBG, Apex                                           |
| Games (MOBA/Other)     | League of Legends, Dota2, Fantasy Westward Journey, Swallow Clouds 16 Rhythms, Genshin Impact |
| Games (Racing)         | KartRider, Forza Horizon 4, QQ Speed                                                          |

# Dataset Construction

## 1. Data Sources

- Video & Film category: Public Kaggle datasets (related to faces, landscapes, sports, animation);
- Game category: Mainly from public Roboflow object detection datasets (labels removed, complete screenshots filtered), with some scarce scenarios (Swallow Clouds 16 Rhythms, QQ Speed, etc.) captured via manual screen recording;
- Office/Industrial Software/Image Editing Software categories: A small amount from the ScreenSpot-Pro dataset on HuggingFace, supplemented mostly by manual screen recording.

Total Dataset Size: 53,438 images, with an average of 2,429 images per category. For detailed category quantities and sources, refer to Chapter 3, Section 3 of the report.

## 2. Core Issues and Solutions

| Core Issue                | Solution                                                                                                                                                                                                                                                                                                                                              |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Validation Set Leakage    | 1. Develop the `dataset_tools` script to deduplicate via hash algorithm (dedupe mode) and filter landscape-oriented images (filter mode); 2. Develop the `video_to_images` script to automatically filter similar frames during capture; 3. Expand the dataset to reduce frame similarity.                                                        |
| Category Imbalance        | 1. Apply color jitter data augmentation to minority classes (without damaging screenshot features); 2. Assign category weights (higher weights for minority classes, lower for majority classes); 3. Fix random seed to 42 for controllable division of training/validation sets; 4. Targetedly expand datasets for low-accuracy categories.          |
| Black Border Interference | 1. Use the trim mode of the `dataset_tools` script to remove black borders from dataset images and retrain; 2. Adjust player aspect ratio to fit the display during real-time inference (e.g., set Bilibili to 16:9); 3. Suggest developing an object detection module in the future to locate effective windows (ROI) and eliminate black borders. |

# Model Architecture and Improvement

## 1. Base Models

- ResNet18: Moderate number of parameters, strong feature extraction capability, fast convergence speed, suitable for edge-side deployment;
- MobileNetV2: Lightweight network with few parameters, core adopts depth-wise separable convolution, focusing on low-power inference.

## 2. Core Innovation: Evolution of Guided CBAM (Convolutional Block Attention Module)

To address the difficulty in distinguishing similar games (e.g., CSGO and CrossFire), inspired by human recognition logic, the model is guided to focus on edge UI elements (gun information, mini-map, etc.) on the screen, avoiding the model "taking shortcuts" by relying on frame style. The evolution is divided into two phases:

1. Initial Scheme (Hard Mask): A 7*7 square-frame mask (1 for edges, 0 for the central 3*3 area) is adopted to force the model to focus on edge UI, solving the confusion of similar games but reducing recognition accuracy for non-similar categories;
2. Improved Scheme (Soft Mask / Adaptive Mask): Adjust the central value of the mask to 0.2 (adaptive learning upper limit of 0.5), balancing edge UI and central frame features. This retains the ability to distinguish similar categories while avoiding recognition errors for non-similar categories, making it the optimal solution.

## 3. Model Training Configuration

| Hyperparameter    | ResNet18 (All Modes)                                                                                    | MobileNetV2 (All Modes)              |
| ----------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| epochs            | 10                                                                                                      | 20                                   |
| batch size        | 32                                                                                                      | 32                                   |
| learning rate     | 0.001                                                                                                   | 0.001                                |
| Training Strategy | Transfer Learning (load pre-trained weights, modify the last layer to adapt to 22-class classification) | Transfer Learning (same as ResNet18) |

# Key Script Description

The core scripts of the project are developed based on PyTorch, covering the entire process of data processing, model training, and inference verification. The specific functions are as follows:

| Script Name                   | Core Function                                                                                                                    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| dataset_tools.py              | Dataset cleaning: dedupe (hash-based deduplication), trim (black border removal), filter (landscape image filtering)             |
| video_to_images.py            | Expand dataset via screen recording screenshots, automatically filter similar frames to reduce validation set leakage risk       |
| train_compare.py              | Base model training and comparison, automatically split training/validation sets (8:2), supporting ResNet18 and MobileNetV2      |
| train_compare_guided.py       | Guided attention mechanism model training, supporting Hard Mask/Soft Mask modes                                                  |
| convert_to_onnx.py            | Convert PTH models to ONNX format, optimize inference speed and reduce video memory usage                                        |
| real_time_inference.py        | Real-time inference for non-guided models, supporting screen capture, result smoothing (majority voting), and UI display         |
| real_time_inference_guided.py | Real-time inference for guided attention models, with the same functions as `real_time_inference.py` and adapted to Mask modes |
| evaluate_accuracy.py          | Model accuracy evaluation, calculating accuracy and confusion matrices for different models and guided strategies                |
| evaluate_speed_benchmark.py   | Model speed benchmark testing, supporting CPU/GPU timing, outputting average latency, standard deviation, and theoretical FPS    |

# Experimental Results and Optimal Solution

## 1. Accuracy Evaluation

- Local Validation Set Accuracy: All models ultimately achieve over 99%, with ResNet18 (Soft Mask) reaching a maximum of 99.89%;
- Practical Evaluation: ResNet18 outperforms MobileNetV2, and the Soft Mask mode effectively distinguishes similar FPS games, reducing confusion errors;
- Comparison with Market Standards: The model's recognition performance is superior to GPT-4V (without local semantics) and close to GPT-4V's (with local semantics) UI recognition level.

## 2. Speed Evaluation

Test Environment: NVIDIA GeForce RTX 3080 (GPU), AMD Ryzen 9 8945HS (CPU). Key conclusions are as follows:

- GPU Environment: ResNet18 (ONNX) achieves an average FPS of 25.73 with a latency of 38.86ms and strong anti-fluctuation capability (only 11x latency difference); MobileNetV2 has no obvious advantages due to operator fragmentation that prevents full utilization of GPU parallel computing capabilities;
- CPU Environment: MobileNetV2 (ONNX) achieves an average FPS of 24.31, adapting to low-power scenarios; ResNet18 (ONNX) reaches an FPS of 20.83 with better stability;
- ONNX Optimization Effect: ResNet18 shows significant speedup (GPU +56%, CPU +82%), and MobileNetV2 achieves a 102% speedup on the CPU side.

## 3. Optimal Solution

ResNet18 + Soft Mask (Guided Attention Mechanism) + ONNX format, balancing accuracy, speed, and stability, adapting to the real-time monitoring requirements of display chip edge-side systems:

- Accuracy: 99.89% (22 categories of scenarios), capable of effectively distinguishing similar games;
- Speed: Average FPS of 25.73 on GPU with latency of 38.86ms, meeting real-time requirements;
- Engineering Adaptation: ONNX format optimizes video memory usage and inference efficiency, adapting to edge-side chip deployment.

# Engineering Implementation Description

## 1. Deployment Preparation

- Model Format: Prioritize ONNX format to reduce redundant computations, improve parallel efficiency, and adapt to edge-side chips;
- Environment Configuration: Install dependent libraries such as PyTorch, OpenCV, mss, NumPy, Qt, adapting to CPU/GPU environments;
- Data Preprocessing: Ensure input images are 224*224 in size, remove black borders, and avoid UI occlusion interference.

## 2. Real-Time Inference Process

1. Run the corresponding inference script (guided/non-guided), specifying the model path and display parameters;
2. The script captures screen frames in real time via the mss library, preprocesses them with OpenCV/NumPy, and inputs them into the model;
3. After model inference, generate attention heatmaps via GradCAM, and smooth results using a sliding window majority voting mechanism;
4. Transmit classification results, confidence, FPS, and heatmaps to the UI interface via Qt signals to complete real-time display.

# Remaining Issues and Future Suggestions

1. Develop an object detection/preprocessing module: Locate effective application windows (ROI) on the screen to automatically eliminate black borders and improve model generalization ability;
2. Expand the dataset: Supplement scarce data such as special animation scenarios, extreme FPS smoke scenarios, game halls/loading interfaces, and enrich long-tail categories;
3. Optimize mask structure: Design adaptive masks based on the UI distribution characteristics of different games to further enhance the ability to distinguish similar categories;
4. Enhance data robustness: Perform occlusion-based data augmentation on game UI to reduce interference from occluders such as mice and progress bars;
5. Explore edge-side adaptive formats: Test the performance of ONNX models on NPU, explore the speedup effect of other model formats (e.g., TensorRT), and adapt to the low-power requirements of display chips.

# Dependencies Description

```markdown
PyTorch == 1.13.0+
OpenCV == 4.7.0+
mss == 9.0.0+
NumPy == 1.24.0+
Qt5 == 5.15.0+
scikit-learn == 1.2.0+
pillow == 9.4.0+
torchvision == 0.14.0+
```

# Notes

All experimental data, training logs, model weights, and demonstration videos of this project can be referred to in the original research report. For script running details and parameter configurations, please check the comments of the corresponding scripts. To further optimize edge-side deployment performance, focus on ONNX format adaptation and NPU deployment testing.

> (Note: Part of the document content may be generated by AI)
>

# 数据集指路：huggingface amazingtrash/scenario-recognition-for-display

# 项目概述

本项目聚焦于显示器芯片端侧AI模型部署，实现对用户屏幕内容的实时识别与分类，核心解决端侧算力受限、类别相似度高、数据集缺失三大核心难点，完成ResNet18与MobileNetV2两种网络在22类场景识别任务中的性能评估与工程化优化，最终实现高精度、低延迟的实时场景识别，适配显示器芯片端侧低功耗、高实时性的部署需求。

核心目标：在显示器芯片端侧部署场景识别模型，支持22类屏幕场景的实时分类，兼顾识别准确度与推理速度，完成从数据集构建、模型改进到工程化落地的全流程开发与验证。

# 核心任务

1. 熟悉ResNet18与MobileNetV2网络结构，掌握原始网络的分类性能基准；
2. 构建22类场景识别数据集，解决数据集缺失、类别不平衡、验证集泄露、黑边干扰等问题；
3. 基于两种骨干网络开展训练与优化，引入引导注意力机制（Guided CBAM）提升相似类别区分能力；
4. 评估模型在准确率、推理速度上的性能，完成模型格式转换（PTH→ONNX）与工程化适配；
5. 开发实时推理脚本与UI交互逻辑，实现屏幕内容的实时截取、推理与结果展示，完成工程化落地验证。

# 场景分类说明

采用扁平化标签处理策略，直接区分22个最底层互斥类别，避免层级推理导致的耗时、权重浪费及连锁错误，具体分类如下（按大类划分）：

| 大类              | 小类（共22类）                                                          |
| ----------------- | ----------------------------------------------------------------------- |
| 影视              | 人脸（Face）、风景（Green Landscape）、体育（Sport）、动画（Animation） |
| 办公              | PPT、Word                                                               |
| 工业软件          | 3DS                                                                     |
| 画图软件          | PS                                                                      |
| 游戏（FPS）       | CSGO、穿越火线、守望先锋、三角洲行动、PUBG、Apex                        |
| 游戏（MOBA/其他） | 英雄联盟、Dota2、梦幻西游、燕云十六声、原神                             |
| 游戏（赛车）      | 跑跑卡丁车、地平线4、QQ飞车                                             |

# 数据集构建

## 1. 数据来源

- 影视类：Kaggle公开数据集（人脸、风景、体育、动画相关）；
- 游戏类：主要来自Roboflow公开目标检测数据集（剔除标签、筛选完整截图），部分稀缺场景（燕云十六声、QQ飞车等）通过手动录屏截取；
- 办公/工业软件/画图软件类：少量来自HuggingFace的ScreenSpot-Pro数据集，大部分通过手动录屏截取补充。

数据集总量：53438张图片，平均每个类别2429张，详细类别数量与来源见报告第三章第三节。

## 2. 核心问题与解决方案

| 核心问题   | 解决方案                                                                                                                                                                                 |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 验证集泄露 | 1. 开发dataset_tools脚本，通过哈希算法去重（dedupe模式）、筛选横屏图片（filter模式）；2. 开发video_to_images脚本，截取时自动过滤相似画面；3. 扩充数据集降低画面相似度。                  |
| 类别不平衡 | 1. 对少量类进行色彩抖动的数据增强（不破坏截图特征）；2. 分配类别权重（少量类权重高、多量类权重低）；3. 固定随机种子42，可控划分训练/验证集；4. 针对性扩充低准确率类别的数据集。          |
| 黑边干扰   | 1. 用dataset_tools脚本的trim模式去除数据集中图片黑边，重新训练；2. 实时推理时，调整播放器长宽比贴合显示器（如B站设置为16:9）；3. 建议后续开发目标检测模块，定位有效窗口（ROI）消除黑边。 |

# 模型架构与改进

## 1. 基础模型

- ResNet18：参数量适中，特征提取能力强，收敛速度快，适合端侧部署；
- MobileNetV2：轻量化网络，参数量少，核心采用深度可分离卷积，主打低功耗推理。

## 2. 核心创新：引导注意力机制（Guided CBAM）演进

针对相似游戏（如CSGO与穿越火线）难以区分的问题，从人类识别逻辑出发，引导模型关注屏幕边缘UI（枪支信息、小地图等），避免模型依赖画面风格“走捷径”，演进分为两个阶段：

1. 初始方案（Hard Mask）：采用7*7回字形掩膜（边缘为1，中心3*3为0），强制模型关注边缘UI，解决相似游戏混淆问题，但会降低非相似类别的识别准确率；
2. 改进方案（Soft Mask / Adaptive Mask）：将掩膜中心值调整为0.2（自适应学习上限0.5），兼顾边缘UI与中心画面特征，既保留相似类区分能力，又避免非相似类识别误差，为最优方案。

## 3. 模型训练配置

| 超参数        | ResNet18（所有模式）                               | MobileNetV2（所有模式） |
| ------------- | -------------------------------------------------- | ----------------------- |
| epochs        | 10                                                 | 20                      |
| batch size    | 32                                                 | 32                      |
| learning rate | 0.001                                              | 0.001                   |
| 训练策略      | 迁移学习（加载预训练权重，修改最后一层适配22分类） | 迁移学习（同ResNet18）  |

# 关键脚本说明

项目核心脚本均基于PyTorch开发，涵盖数据处理、模型训练、推理验证全流程，具体功能如下：

| 脚本名称                      | 核心功能                                                                  |
| ----------------------------- | ------------------------------------------------------------------------- |
| dataset_tools.py              | 数据集清洗：dedupe（哈希去重）、trim（去除黑边）、filter（筛选横屏图片）  |
| video_to_images.py            | 录屏截图扩充数据集，自动过滤相似画面，降低验证集泄露风险                  |
| train_compare.py              | 基础模型训练与对比，自动划分训练/验证集（8:2），支持ResNet18与MobileNetV2 |
| train_compare_guided.py       | 引导注意力机制模型训练，支持Hard Mask/Soft Mask两种模式                   |
| convert_to_onnx.py            | 将PTH模型转换为ONNX格式，优化推理速度、减少显存占用                       |
| real_time_inference.py        | 无引导模型的实时推理，支持屏幕截取、结果平滑（多数投票）、UI展示          |
| real_time_inference_guided.py | 引导注意力模型的实时推理，功能同real_time_inference.py，适配Mask模式      |
| evaluate_accuracy.py          | 模型准确率评估，计算不同模型、不同引导策略的准确率与混淆矩阵              |
| evaluate_speed_benchmark.py   | 模型速度基准测试，支持CPU/GPU计时，输出平均延迟、标准差、理论FPS          |

# 实验结果与最优方案

## 1. 准确率评估

- 本地验证集准确率：所有模型最终均达到99%以上，ResNet18（Soft Mask）最高达99.89%；
- 实操评估：ResNet18性能优于MobileNetV2，Soft Mask模式能有效区分相似FPS游戏，减少混淆误差；
- 对比市面标准：模型识别性能优于GPT-4V（无本地语义），接近GPT-4V（有本地语义）的UI识别水平。

## 2. 速度评估

测试环境：NVIDIA GeForce RTX 3080（GPU）、AMD Ryzen 9 8945HS（CPU），核心结论如下：

- GPU环境：ResNet18（ONNX）平均FPS达25.73，延迟38.86ms，抗波动能力强（延迟落差仅11倍）；MobileNetV2优势不明显，因算子碎片化无法充分利用GPU并行能力；
- CPU环境：MobileNetV2（ONNX）平均FPS达24.31，适配低功耗场景；ResNet18（ONNX）FPS达20.83，稳定性更优；
- ONNX优化效果：ResNet18提速显著（GPU+56%、CPU+82%），MobileNetV2在CPU端提速达102%。

## 3. 最优方案

ResNet18 + Soft Mask（引导注意力机制）+ ONNX格式，兼顾准确率、速度与稳定性，适配显示器芯片端侧实时监测需求：

- 准确率：99.89%（22类场景），能有效区分相似游戏；
- 速度：GPU端平均FPS 25.73，延迟38.86ms，满足实时性要求；
- 工程适配：ONNX格式优化显存占用与推理效率，适配端侧芯片部署。

# 工程化落地说明

## 1. 部署准备

- 模型格式：优先使用ONNX格式，减少冗余计算、提升并行效率，适配端侧芯片；
- 环境配置：安装PyTorch、OpenCV、mss、NumPy、Qt等依赖库，适配CPU/GPU环境；
- 数据预处理：确保输入图片为224*224尺寸，去除黑边，避免UI遮挡干扰。

## 2. 实时推理流程

1. 运行对应推理脚本（有引导/无引导），指定模型路径与显示器参数；
2. 脚本通过mss库实时截取屏幕画面，经OpenCV/NumPy预处理后输入模型；
3. 模型推理完成后，通过GradCAM生成注意力热力图，采用滑动窗口多数投票机制平滑结果；
4. 通过Qt信号将分类结果、置信度、FPS、热力图传递至UI界面，完成实时展示。

# 遗留问题与后续建议

1. 开发目标检测/预处理模块：定位屏幕有效应用窗口（ROI），自动消除黑边，提升模型泛化能力；
2. 扩充数据集：补充动漫特殊场景、FPS烟雾极端场景、游戏大厅/loading界面等稀缺数据，丰富长尾类别；
3. 优化掩膜结构：针对不同游戏的UI分布特点，设计自适应掩膜，进一步提升相似类别区分能力；
4. 增强数据鲁棒性：对游戏UI进行遮挡类数据增强，降低鼠标、进度条等遮挡物的干扰；
5. 探索端侧适配格式：测试ONNX模型在NPU上的性能，探索其他模型格式（如TensorRT）的提速效果，适配显示器芯片低功耗需求。

# 依赖库说明

```markdown
PyTorch == 1.13.0+
OpenCV == 4.7.0+
mss == 9.0.0+
NumPy == 1.24.0+
Qt5 == 5.15.0+
scikit-learn == 1.2.0+
pillow == 9.4.0+
torchvision == 0.14.0+
```

# 备注

本项目所有实验数据、训练日志、模型权重、演示视频可参考原研究报告，脚本运行细节与参数配置可查看对应脚本注释，如需进一步优化端侧部署性能，可重点关注ONNX格式适配与NPU部署测试。

> （注：文档部分内容可能由 AI 生成）
