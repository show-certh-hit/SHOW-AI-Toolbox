# Accident Detection
This model is a lightweigted one for the detection of tha accident based on GPS data. This PoC points out that the widely available GPS data can offer significant insights on the accident detection domain. 
### Introduction

The main goal is the identification of an accident data via the anomaly detection applied in GPS data (timestamo, latitude, longitude, speed and acceleration)
### Pre-requisites

##### Tensorflow 1.14
##### Opencv
##### Matplotlib
##### Numpy

### Methodology
![Alt text](./img/methodology.png "Optional title")


### Dataset & Features

The model has been trained and evaluated based on the GPS data of an operating SHOW site. The data will be circulated via Zenodo.
### Usage

#### Data Generator for accident cases
```
python3 data_generator.py
```

#### Training & Evaluation
```
python3 full_accident_detection.py
```


### Citing

PoC has been published as Papadopoulos, A., Sersemis, A., Spanos, G., Lalas, A., Liaskos, C., Votis, K., & Tzovaras, D. (2024). Lightweight accident detection model for autonomous fleets based on GPS data. Transportation research procedia, 78, 16-23.

### Contact
Alexandros Papadopoulos. alexpap@iti.gr
