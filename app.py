

import os, io, base64, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.utils import to_networkx
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import LinearRegression

from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# GCN Encoder (same architecture as the notebook)
# ---------------------------------------------------------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# Core analysis pipeline

FEATURES = [
    "Number of Services",
    "Number of Medicare Beneficiaries",
    "Number of Distinct Medicare Beneficiary/Per Day Services",
    "Average Medicare Allowed Amount",
    "Average Submitted Charge Amount",
    "Average Medicare Payment Amount",
    "Average Medicare Standardized Amount",
]

PRESENTATION_COLS = [
    "National Provider Identifier",
    "Last Name/Organization Name of the Provider",
    "First Name of the Provider",
    "Provider Type",
    "State Code of the Provider",
    "Number of Services",
    "Average Medicare Allowed Amount",
    "Average Medicare Payment Amount",
]


def run_pipeline(csv_bytes: bytes, n_top: int = 5, sample_size: int = 10000,
                 k: int = 5, latent_dim: int = 16, epochs: int = 200):
    """Run the full fraud-detection pipeline and return result dicts."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load & clean -------------------------------------------------------
    df = pd.read_csv(io.BytesIO(csv_bytes))
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").apply(
                pd.to_numeric, errors="coerce"
            )
    df_clean = df.dropna(subset=FEATURES).copy()

    # --- Feature Transformation (Log1p) to handle skew ---
    df_log = df_clean.copy()
    for col in FEATURES:
        df_log[col] = np.log1p(df_clean[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log[FEATURES])
    x = torch.tensor(X_scaled, dtype=torch.float)

    # --- Sampling -----------------------------------------------------------
    if len(df_clean) > sample_size:
        indices = np.random.choice(len(df_clean), sample_size, replace=False)
        x_sample = x[indices]
        df_sample = df_clean.iloc[indices].reset_index(drop=True)
    else:
        x_sample = x
        df_sample = df_clean.reset_index(drop=True)

    # --- KNN graph ----------------------------------------------------------
    adj = kneighbors_graph(x_sample.numpy(), k, mode="connectivity", include_self=False)
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    data = Data(x=x_sample, edge_index=edge_index).to(device)

    # --- Model --------------------------------------------------------------
    encoder = GCNEncoder(in_channels=data.x.shape[1], out_channels=latent_dim)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

    # --- Anomaly scores -----------------------------------------------------
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    z_np = z.cpu().numpy()
    x_np = data.x.cpu().numpy()

    decoder = LinearRegression().fit(z_np, x_np)
    x_recon = decoder.predict(z_np)
    
    # Weighted Anomaly Score: Prioritize payment anomalies (indices 3-6) over volume (0-2)
    # 0: Services, 1: Beneficiaries, 2: Daily Distinct Benes
    # 3: Allowed Amt, 4: Charge Amt, 5: Payment Amt, 6: Std Payment Amt
    mse_per_feature = (x_np - x_recon) ** 2
    weights = np.array([0.1, 0.1, 0.1, 1.5, 1.5, 1.5, 1.5])
    weighted_mse = mse_per_feature * weights

    df_sample["Anomaly_Score"] = np.mean(weighted_mse, axis=1)

    top_anomalies = df_sample.sort_values("Anomaly_Score", ascending=False)

    # --- Top N anomalies table ----------------------------------------------
    available_cols = [c for c in PRESENTATION_COLS if c in top_anomalies.columns]
    top_n = top_anomalies[available_cols].head(n_top).copy()

    def fmt(row):
        first = row.get("First Name of the Provider", "")
        last  = row.get("Last Name/Organization Name of the Provider", "")
        if pd.isna(first) or first == "":
            return str(last)
        return f"{first} {last}"

    top_n["Formatted Name"] = top_n.apply(fmt, axis=1)
    top_n_display = top_n[["Formatted Name", "Provider Type",
                            "State Code of the Provider",
                            "Average Medicare Payment Amount"]].copy()
    top_n_display.columns = ["Name", "Provider Type", "State", "Avg Medicare Payment"]
    top_n_html = top_n_display.to_html(index=False, classes="table table-striped table-hover",
                                        border=0)

    # --- Comparison table ---------------------------------------------------
    avg_stats = df_sample[FEATURES].mean()
    top1_stats = top_anomalies.iloc[0][FEATURES]
    comp = pd.DataFrame({
        "Metric": FEATURES,
        "Average Provider": avg_stats.values,
        "Top Suspicious Provider": top1_stats.values,
    })
    comp["Deviation (× Normal)"] = (comp["Top Suspicious Provider"] /
                                     comp["Average Provider"]).round(2)
    comp_html = comp.round(2).to_html(index=False, classes="table table-striped table-hover",
                                       border=0)

    # --- Graph 1: Anomaly score distribution --------------------------------
    fig1, ax = plt.subplots(figsize=(10, 5))
    scores = df_sample["Anomaly_Score"]
    threshold = scores.mean() + 2 * scores.std()
    n_above = int((scores > threshold).sum())
    n_total = len(scores)

    sns.histplot(scores, bins=50, kde=False, color="skyblue", edgecolor="white",
                 alpha=0.7, ax=ax, label="Count per bin")
    ax2 = ax.twinx()
    sns.kdeplot(scores, color="navy", linewidth=2, ax=ax2, label="KDE")
    ax2.set_ylabel("")
    ax2.set_yticks([])
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold (μ+2σ = {threshold:.4f})")

    ax.set_title("Anomaly Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Anomaly Score (Reconstruction Error)")
    ax.set_ylabel("Number of Providers")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    dist_img = _fig_to_b64(fig1)

    # --- Graph 2: Network graph ---------------------------------------------
    data_cpu = data.cpu()
    G_full = to_networkx(data_cpu, to_undirected=True)

    anomaly_indices = top_anomalies.index[:30].tolist()
    anomaly_set = set(anomaly_indices)
    neighbors = set()
    for idx in anomaly_indices:
        if idx in G_full:
            neighbors.update(G_full.neighbors(idx))
    subset = list(anomaly_set | neighbors)
    G_sub = G_full.subgraph(subset).copy()

    all_scores = df_sample["Anomaly_Score"]
    snorm = Normalize(vmin=all_scores.min(), vmax=all_scores.max())
    node_colors, node_sizes, labels = [], [], {}
    for node in G_sub.nodes():
        sc = df_sample.loc[node, "Anomaly_Score"]
        if node in anomaly_set:
            node_colors.append(cm.Reds(snorm(sc)))
            node_sizes.append(200)
            r = anomaly_indices.index(node) + 1 if node in anomaly_indices[:10] else None
            if r:
                pt = str(df_sample.loc[node, "Provider Type"])
                labels[node] = f"#{r}: {pt[:18]}..." if len(pt) > 18 else f"#{r}: {pt}"
        else:
            node_colors.append("#87CEFA")
            node_sizes.append(40)

    fig2, ax3 = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G_sub, seed=42, k=0.3, iterations=80)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.15, edge_color="gray", ax=ax3)
    nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, edgecolors="black", linewidths=0.5, ax=ax3)
    label_pos = {k: (v[0], v[1] + 0.04) for k, v in pos.items()}
    nx.draw_networkx_labels(G_sub, label_pos, labels, font_size=8,
                            font_weight="bold", ax=ax3)
    red_p = mpatches.Patch(color="red", label="Suspicious (High Error)")
    blue_p = mpatches.Patch(color="#87CEFA", label="Normal (Low Error)")
    ax3.legend(handles=[red_p, blue_p], loc="upper left", fontsize=10)
    ax3.set_title("Provider Similarity Network — Anomaly Detection",
                  fontsize=14, fontweight="bold")
    ax3.axis("off")
    plt.tight_layout()
    net_img = _fig_to_b64(fig2)

    # --- Training loss graph ------------------------------------------------
    fig3, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(losses, color="#2980b9", linewidth=1.5)
    ax4.set_title("Training Loss (Reconstruction Error)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.grid(True, alpha=0.2)
    plt.tight_layout()
    loss_img = _fig_to_b64(fig3)

    return {
        "top_n_html": top_n_html,
        "comp_html": comp_html,
        "dist_img": dist_img,
        "net_img": net_img,
        "loss_img": loss_img,
        "n_total": n_total,
        "n_above": n_above,
        "threshold": round(float(threshold), 4),
        "n_top": n_top,
    }


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")



# Flask app

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    n_top = int(request.form.get("n_top", 5))
    n_top = max(1, min(n_top, 100))

    csv_bytes = file.read()
    try:
        results = run_pipeline(csv_bytes, n_top=n_top)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
