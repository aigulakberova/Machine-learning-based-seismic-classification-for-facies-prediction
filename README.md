# Machine-learning-based-seismic-classification-for-facies-prediction
#### This is a repository for the thesis 'Machine-learning-based-seismic-classification-for-facies-prediction'.

# Abstract
This thesis explores the performance of machine learning (ML) methods for 
predicting facies from seismic attributes for 2D and 3D datasets. 
It focuses on building, training, and testing four supervised methods: Logistic Regression, Support Vector Machines, 
K-Nearest Neighbors, and Random Forest; and one deep learning method: Neural Network with two hidden layers. 
A realistic synthetic facies model with complex depositional systems, and a synthetic seismic cube from the facies model are used for the 
comparison of facies prediction performed by the ML approach with the ground-truth facies distribution. 
This comparison makes it possible to validate the ML models’ prediction based on wells and seismic. 
In addition, the research evaluates the role of the number of wells and their locations, the impact of seismic data frequency, 
and the effect of using various seismic attributes. The most important features for facies prediction are seismic inversion and 
relative acoustic impedance. Instantaneous frequency and envelope have little effect on the accuracy of the ML prediction. 
Incorporating information about the lateral geometry of the facies in the reservoir also improves the accuracy of the ML prediction.

# Key words

Machine learning,
facies,
synthetic models,
2D facies prediction,
3D facies prediction,
seismic attributes,
seismic inversion

# Why it is useful
This repository can be used for predicting facies distribution in 2D sections and 3D cubes, including thick and conformable layers and interbedded thin layers of facies. 

# Content
Four different cases were considered in this thesis. The folders contain SEGY files with seismic attributes, seismic inversion for every case. 

Case 1 - 2D section with thick and conformable layers.

Case 2 - 2D section with thin layers of facies.

Case 3 - 2D section with thick and conformable layers with a normal fault.

Case 4 - 3D case.


# Who wrote this
Aigul T. Alvi

The thesis is done by the supervision of Nestor Cardozo (UiS) and Lothar Schulte (SLB).

# Required Python libraries
Matplotlib    3.4.3

NumPy         1.22.4

Scikit Learn  0.24.2

TensorFlow    2.12.0

Pandas        1.3.4

Plotly        5.5.0

Seaborn       0.11.2

Segyio        1.9.10

Segysak       0.3.4

# Contact for further information
akberovaash@gmail.com
