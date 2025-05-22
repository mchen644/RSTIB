# Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official code for the paper "**Information Bottleneck-guided MLPs for Robust Spatial-temporal Forecasting**" (ICML 2025).

* Authors: Min Chen, [Guansong Pang](https://sites.google.com/site/gspangsite), [Wenjun Wang](http://www.smartsafety.cn/wiki/index.php/WangWenjun) and [Cheng Yan](https://yancheng-tju.github.io/yancheng.github.io/)

## Overview

Spatial-temporal forecasting (STF) plays a pivotal role in urban planning and computing. Spatial-Temporal Graph Neural Networks (STGNNs) excel in modeling spatial-temporal dynamics, thus being robust against noise perturbation. However, they often suffer from relatively poor computational efficiency. Simplifying the architectures can speed up these methods but it also weakens the robustness w.r.t. noise interference. In this study, we aim to investigate the problem -- can simple neural networks such as Multi-Layer Perceptrons (MLPs) achieve robust spatial-temporal forecasting yet still be efficient? To this end, we first disclose the dual noise effect behind the spatial-temporal data noise, and propose theoretically-grounded principle termed Robust Spatial-Temporal Information Bottleneck (RSTIB) principle, which preserves wide potentials for enhancing the robustness of different types of models. We then meticulously design an implementation, termed RSTIB-MLP, along with a new training regime incorporating a knowledge distillation module, to enhance the robustness of MLPs for STF while maintaining its efficiency. Comprehensive experimental results show that an excellent trade-off between the robustness and the efficiency can be achieved by RSTIB-MLP compared to state-of-the-art STGNNS and MLP models.
