# NDRGC-TCN
This repository contains the model implementation of our proposed NDRGC-TCN network framework, along with the self-developed tools, baseline model codes, comparative experiments, ablation studies, and partial data visualization scripts.
Due to the excessively large size of the log files generated during the experiments, we have only uploaded the training logs for the proposed network and its ablation variants. If researchers require the complete log files of the baseline models, please feel free to contact the authors via email.
## Framework Figure
<img width="1687" height="860" alt="主图" src="https://github.com/user-attachments/assets/c3f36969-fd22-4e4f-a1f5-fc213a3754e3" />

## Fusion Details Figure
<img width="1326" height="665" alt="PM-COH细节图1" src="https://github.com/user-attachments/assets/fa28e1ed-05d7-4488-8a67-6632b39cd58f" />

## Experimental Environment and Dependencies

**Python version**: 3.10.2  
**PyTorch version**: 2.5.1+cpu  
**Braindecode version**: 0.8  
**MNE version**: 1.9.0  
**MNE-Connectivity version**: 0.7.0 (for PLV and COH calculations)  
**EEGLAB version**: 2026.0.0 (MATLAB toolbox, used for data preprocessing and validation)  
**Pandas version**: 2.2.2  
**NumPy version**: 1.24.3  
**Matplotlib version**: 3.9.2  
**Scikit-learn version**: 1.7.2  
**Seaborn version**: 0.13.2  
**SciPy version**: 1.15.3  
**Thop version**: 0.1.1.post2209072238 (for model parameters and FLOPs calculation)  
**PyTorch Grad-CAM version**: 1.5.0 (for CAM activation feature visualization)  

**Computing Platform**: NVIDIA GeForce RTX 4090 GPU
