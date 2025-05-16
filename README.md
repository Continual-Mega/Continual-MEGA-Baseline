# Continual-Mega: A Large-scale Benchmark for Generalizable Continual Anomaly Detection

This repository contains the evaluation code for the Continual-Mega benchmark, submitted to NeurIPS 2025. \
We provide the checkpoint files of the proposed model along with the evaluation code for the Continual-Mega benchmark. \
The training code will be made publicly available at a later date.

## Datasets
### Download
The datasets are available through Hugging Face. \
https://huggingface.co/datasets/Continual-Mega/Continual-Mega-Neurips2025

### Structure
The datasets directory should be structured as follows:
``` 
data/
├── continual_ad/              # Our proposed ContinualAD dataset
├── mvtec_anomaly_detection/   # MVTec-AD dataset
├── VisA_20220922/             # VisA dataset
├── VIADUCT/                   # VIADUCT dataset 
├── Real-IAD-512/              # RealIAD dataset (512 size)
├── MPDD/                      # MPDD dataset
└── BTAD/                      # BTAD

``` 

## Evaluation
### checkpoint files
We provide checkpoint files for the setting with 30 classes per task in Scenario 2.

### Continual Settings
``` 
sh eval_continual.sh
``` 
### Zero-Shot Generalization
``` 
sh eval_zero.sh
``` 