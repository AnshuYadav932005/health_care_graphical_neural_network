# Learning Checklist: specific-gnn-fraud-detection.md

## Phase 1: Data Engineering & Logic (The "Why")
- [ ] **Logarithmic Scaling**: Explain why we used `np.log1p` on the "Number of Services" column. What would happen to the Euclidean distance without it?
- [ ] **Weighted Error**: We manually multiplied the reconstruction error for payment columns by 1.5. Why does a standard Mean Squared Error (MSE) not capture "domain knowledge" by default?
- [ ] **Graph Construction**: We used K-Nearest Neighbors (KNN) to build the graph. What determines if two providers are connected?

## Phase 2: The GNN Model (The "How")
- [ ] **Encoder vs. Decoder**: In `GAE(encoder)`, what is the specific job of the encoder? What is the decoder doing in a standard GAE?
- [ ] **Latent Space**: The model compresses 7 features into 16 dimensions. Why do we need this "bottleneck"? Why not just compare the raw rows?
- [ ] **Reconstruction Loss**: We plot a loss curve. What does it mean if the curve goes down? What does it mean if it stays flat?

## Phase 3: Web Engineering (The "Interface")
- [ ] **Flask Request Cycle**: When you click "Run Analysis", how does the data get from the HTML form to the Python tensor and back?
- [ ] **Statelessness**: The `app.py` loads the model and runs it on every request. What is the trade-off of this vs. keeping a model loaded in memory?
