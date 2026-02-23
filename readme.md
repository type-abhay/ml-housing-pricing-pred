# California Housing Price Predictor

## 1. About the Project
This project is an end-to-end Machine Learning web application designed to predict real estate valuations based on the 1990 California census demographic data. Built as a comprehensive academic pipeline, it explores multiple facets of machine learning paradigms including Regression, Classification, Support Vector Machines, and Neural Networks.

The target variable, Median House Value (`MedHouseVal`), is evaluated both continuously (via Regression) and categorically by separating the market into Low, Medium, and High tiers. The finalized models are served through a sleek, modern, dark-mode web interface built on Flask, allowing for real-time inference.

## 2. Working
The architecture is divided into two distinct components: the rigorous analytical pipeline and the web deployment backend.

* **Data Processing:** The dataset undergoes a strict, reproducible $70/15/15$ split (Training, Validation, Testing). Feature scaling is performed strictly using the training distribution to prevent data leakage.
* **The Machine Learning Pipeline (`ml_pipeline.py`):** * Computes Exploratory Data Analysis (EDA) visualizations directly to a `plots/` directory.
  * Trains and evaluates Simple and Multiple Linear Regression models.
  * Classifies derived market tiers using Logistic Regression, Decision Trees, and Random Forests.
  * Implements a Support Vector Machine with an RBF kernel.
  * Trains a custom Multi-Layer Perceptron (Neural Network) featuring explicit early stopping monitored against the validation set.
  * Serializes the highest performing models and scalers into a `models/` directory using `joblib`.
* **The Web Application (`app.py`):** A Flask-based backend that loads the pre-trained models. It ingests user parameters from an HTML/CSS frontend, applies the saved scaling transformations, and dynamically returns the predicted house value.

## 3. How to Run It on Your Computer
Follow these exact steps to initialize the environment, train the models, and launch the web application.

### Prerequisites
Ensure you have Python 3.8+ installed on your system. 

### Step 1: Install Dependencies
Open your terminal, navigate to the project directory, and install the required packages:
`pip install -r requirements.txt`

### Step 2: Execute the ML Pipeline
Before running the web server, you must train the models and generate the required analytical plots. Run the following command:
`python ml_pipeline.py`
*(Note: This will automatically create `models/` and `plots/` directories in your folder, saving the `.pkl` files and `.png` visualizations.)*

### Step 3: Launch the Flask Server
Once the pipeline finishes and the models are safely stored, ignite the backend server:
`python app.py`

### Step 4: Access the Web App
Open your favorite web browser and navigate to the local host address provided in your terminal (typically `http://127.0.0.1:5000/`). Enter the district demographics into the sleek dark-mode interface and click "Predict Value" to test the model in real-time.