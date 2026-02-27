# Healthcare Fraud Detection using Unsupervised Graph Neural Networks (GNN)

This project implements an unsupervised Graph Neural Network (specifically a Graph Autoencoder) to detect anomalous healthcare providers who may be committing fraud. The model learns normal patterns of behavior from the data and flags providers that deviate significantly from these patterns.

## Project Structure

- `healthcare_fraud_detection.ipynb`: The main Jupyter Notebook containing the entire pipeline:
  - Data Loading & Preprocessing
  - Graph Construction (KNN)
  - GNN Model Definition (Graph Autoencoder)
  - Training Loop
  - Anomaly Detection (Reconstruction Error)
  - Visualization
- `Healthcare Providers.csv`: The dataset used for training and analysis.
- `requirements.txt`: List of Python libraries required to run the project.

## How to Run

1. **Install Dependencies**:
   Open a terminal in this folder and run:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you have a GPU, ensure you install the correct version of PyTorch compatible with your CUDA version.*

2. **Open the Notebook**:
   Open `healthcare_fraud_detection.ipynb` in VS Code or Jupyter Lab.

3. **Execute the Code**:
   Run all cells in order. The notebook will:
   - Load the data.
   - Build a graph where providers are nodes connected by similarity.
   - Train the model to reconstruct this graph.
   - Output a list of suspicious providers (anomalies).
   - Display a visualization of the anomalies vs. normal providers.

## Methodology

1. **Graph Construction**: We treat each provider as a node. Edges are created between providers with similar behavior (services, payments) using K-Nearest Neighbors (KNN).
2. **Graph Autoencoder (GAE)**: A GCN encoder compresses the provider's features and connections into a low-dimensional latent space.
3. **Anomaly Detection**: We attempt to reconstruct the original features from this latent space. Providers with high reconstruction errors (high difference between original and reconstructed data) are flagged as anomalies.
