# Multi Algorithm Classification for Daily and Sports Activity Recognition using Inertial Sensors

## Introduction
This project investigates the classification of 19 daily and sports activities using inertial sensor data. The goal is to develop an accurate classification model capable of distinguishing between dynamic and stationary activities, and to evaluate the performance of various Machine Learning algorithms in the context of sensor-based human activity recognition.

## Problem Statement
Inertial sensors (accelerometers, gyroscopes) are fundamental in many applications, from navigation  to health monitoring. However, challenges such as signal noise  and the complexity of multi-sensor data fusion  can reduce classification accuracy, particularly for subtle movements or stationary activities. This project aims to address the challenge of classifying both stationary and dynamic activities from noisy inertial sensor data, where similar movement patterns often hinder accuracy for real-time applications.

## Dataset
Dataset: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities
The primary data used in this study comes from the **"Daily and Sports Activities"** dataset, available on the UCI Machine Learning Repository.
* **Number of Activities**: 19 types of daily and sports activities (e.g., walking, jumping, sitting, standing).
* **Subjects**: Collected from eight subjects (four male and four female, aged 20-30 years).
* **Recording**: Each activity was recorded for five minutes at a sampling frequency of 25 Hz.
* **Sensors**: The dataset includes 45 channels of multi-sensor data originating from five Xsens MTx units. These sensor units were positioned on various body parts: chest, waist, right arm, left arm, and right leg. Each Xsens MTx unit recorded 9-axis data (three-axis accelerometer, three-axis gyroscope, three-axis magnetometer).

## Methodology
This research adopts both *supervised learning* and *unsupervised learning* approaches through the following stages:

1.  **Data Acquisition & Preprocessing**:
    * Raw data collected from UCI `.txt` files.
    * Outlier removal (values exceeding three times the standard deviation from the mean).
    * Data normalization using `StandardScaler`.
    * Time window segmentation into 2.5-second segments with 50% overlap.
2.  **Feature Extraction**:
    * From each processed segment, 180 statistical features were extracted. These include Mean, Standard Deviation, Skewness, and Kurtosis for each of the 45 sensor channels.
3.  **Data Splitting**:
    * **Training Data**: From subjects p1-p5, resulting in 5700 segments.
    * **Testing Data**: From subjects p6-p8, resulting in 3420 segments.
4.  **Model Building & Training**: Three main classification models were built and evaluated:
    * **Random Forest (RF)**: Configured with `n_estimators=500` and `random_state=42`.
    * **Support Vector Machine (SVM)**: Used a Radial Basis Function (RBF) kernel with `random_state=42`. Hyperparameters C and gamma were optimized.
    * **Neural Network (NN)**: A Sequential model with two hidden layers (128 and 64 neurons), ReLU activation, and Dropout (ratio 0.3) for regularization. The output layer had 19 neurons with Softmax activation.
    * **Principal Component Analysis (PCA)**: Used for dimensionality reduction (to 2 components) and data visualization.

## Results and Discussion

### Comparative Model Performance
![image](https://github.com/user-attachments/assets/36dace33-c38a-4d7e-a8b5-7947005780d5)

The comparison of test accuracies among the models showed significant performance variations:
* **Random Forest**: **0.9146** (highest accuracy).
* **Support Vector Machine (SVM)**: 0.8418.
* **Neural Network**: 0.7281.

Random Forest demonstrated superior performance, consistent with its capability in handling high-dimensional data and its robustness to noise.

### Challenges in Classifying Stationary Activities
While the models generally showed strong performance on dynamic activities, significant challenges were found in classifying stationary activities.
* For example, Random Forest achieved a recall of only **0.16** for the "standing" activity, indicating the model's difficulty in identifying most instances of this activity.
* SVM performed even worse for "standing" with a recall of 0.04.
* The Neural Network showed significant overfitting, with training accuracy approaching perfect while validation accuracy remained low (~0.20-0.22).

Feature analysis and PCA indicated that time-domain features might not be sufficient to distinguish subtle movements. PCA's two principal components explained only 23.52% of the total data variance, suggesting a higher intrinsic dimensionality of the multi-sensor data.

## Conclusion
This study successfully explored and evaluated the performance of various Machine Learning models for activity classification using inertial sensors. Random Forest showed the best overall performance, but the classification of stationary activities remains a significant challenge. This indicates the need for more sophisticated features from the frequency domain or more complex model architectures to capture subtle differences in low-amplitude movements.

## Recommendations & Future Work
1.  **Advanced Feature Exploration**: Integrating features from the frequency domain (e.g., Power Spectral Density) or wavelet-based features to improve discrimination of stationary activities.
2.  **Advanced Models**: Trying deeper Neural Network architectures (e.g., Convolutional Neural Networks or Recurrent Neural Networks) that can intrinsically learn temporal and spatial features from raw sensor data, provided a larger dataset is available.
3.  **Class Imbalance Handling**: If there are activities with imbalanced data, techniques like SMOTE (Synthetic Minority Over-sampling Technique) can be applied.
4.  **Model Personalization**: Developing models that can adapt to individual movement patterns to enhance accuracy, especially in personal health applications.

## Technologies Used
* Python 3.9
* Scikit-learn
* TensorFlow
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook

## How to Run the Project (Example)
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Zenn711/classification-of-daily-activities-and-sports.git
    cd your-repo-name
    ```
2.  **Download Dataset**:
    * The "Daily and Sports Activities" dataset can be downloaded from the UCI Machine Learning Repository.
    * Ensure the data folder structure matches what the script expects (e.g., `data/a01/p01/s01.txt`). You will need to set the `base_path` variable in the code to your dataset's location.
3.  **Install Dependencies**:
    You will need to install the following Python libraries:
    ```bash
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
    ```
4.  **Run Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
    Open your project notebook file (e.g., `HAR_Project.ipynb`) and run the cells sequentially.

## Author
Muhammad Harits Naufal
Department of Electrical Engineering and Informatics, Vocational School, Universitas Gadjah Mada

---
