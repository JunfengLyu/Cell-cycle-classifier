# Cell_cycle_distinguisher

A project for cell cycle stage classification and analysis using microscopy images, image processing, and deep learning models.

## Overview

This repository contains code and data-processing pipelines for:

- manual or semi-automatic cell cropping
- dataset construction from raw microscopy images
- cell segmentation and candidate detection
- cell cycle stage classification
- benchmarking of deep learning models
- feature extraction and manifold learning visualization
- application scripts for trained models on new images

The project mainly focuses on classifying cells into the following stages:

- Interphase (I)
- Prophase (P)
- Metaphase (M)
- Anaphase (A)
- Telophase (T)

## Project structure

```text
Cell_cycle_distinguisher/
├── Dataset_raw/              # raw microscopy images
├── Dataset_20times/          # cropped dataset at 20x magnification
├── Dataset_100times/         # cropped dataset at 100x magnification
├── Output/                   # training outputs, figures, tables, saved models
├── Application/              # prediction results on new data
├── Cellpose/                 # Cellpose-related outputs or intermediate files
├── *.py                      # training, application, preprocessing scripts
└── README.md

## Demo

![Example prediction](Application/100times_results/visualizations/v1_prediction.png)
