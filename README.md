# 基于深度学习的细胞周期图像识别器

本项目系北京大学2026春**细胞生物学实验**上午班第三组完成的自主实验，组员LJF, GYC, HZX, XYX。本实验采集两个班级收集的**免疫荧光标记Tubulin的HeLa细胞**和**Feulgen法染色DNA的蚕豆根尖细胞**图像，训练深度网络进行图像分割和识别。

## 概览

本项目的代码功能包括：

- cellcrop: 手动分割和标记细胞周期的数据集架构代码，包含早期Matlab版本
- training: 利用手动分割细胞图像训练细胞周期识别神经网络，主要架构为ResNet18与ResNet34
- manifold: 对训练得ResNet34对HeLa 100倍率细胞的深度表征空间进行无监督流形降维学习
- cellpose: 使用人工半监督训练方法手动微调本地部署的cellpose以分割后/末期细胞
- application: 最终应用微调cellpose模型分割图版，并用训练模型判断细胞周期的应用代码



## Demo

![Example prediction](Application/100times_results/visualizations/v1_prediction.png)


## 致谢

- 感谢全体同学贡献数据集
- 感谢课程组教师和助教给予湿实验帮助和完成初步数据收集工作
- 感谢CXR学长参与讨论cellpose微调与人工guidance建议
- 感谢LBR同学讨论图像分割算法
