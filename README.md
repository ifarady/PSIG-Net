# PSIG-Net: Pseudo Skin Image Generation Network

This repository contains the implementation code for our latest work, PSIG-Net, a two-stage framework for skin cancer classification. PSIG-Net addresses the challenge of limited training data and potential outliers by leveraging two key components:

  

### Current Status:

Our paper describing PSIG-Net is currently under review in [Biomedical Signal Processing and Control](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control). We will be actively maintaining and updating this repository with future improvements and insights.

![PSIG-Net Image](https://github.com/ifarady/PSIG-Net/blob/main/psig-net.png)

## Stage-1: Pseudo Sample Generation

PSIG-Net generates high-quality pseudo samples to augment the training data, enriching the representation space and improving model robustness.

  

## Stage-2: Outlier Controlling

A robust loss function and data filtering mechanism are employed to identify and mitigate the impact of outliers, further enhancing the model's generalization ability.

 
### Highlights:

- **Improved Accuracy:** PSIG-Net achieves state-of-the-art performance on skin cancer classification benchmarks on ISIC-2017 and ISIC-2018 datasets

- **Data Efficiency:** It effectively utilizes limited training data by generating informative pseudo samples using a GAN-based model.

- **Outlier Resilience:** The modification of the Siamese-based network is robust to outliers, leading to more reliable and generalizable models.

### Get Involved:

  
Feel free to explore the code, experiment, and contribute to the project! We welcome feedback and suggestions.

References: If you use or refer to PSIG-Net in your work, please cite the following: 
