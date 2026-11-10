# %% [markdown]
# # Lab 5: Decision Tree Classifier & Model Evaluation
# **Course:** IC-709D | AI Lab  **Unit 5 (CO4, CO5)**
#
# **Activities:** Student Placement Prediction, Disease Prediction,
# Loan Approval Prediction, Decision Tree Visualization.
# **Application:** Intelligent Decision Support System.
# **Analysis:** Evaluate Decision Tree performance, compare evaluation metrics.

# %% [code] imports
"""Lab 5 - Decision Tree Classifier & Model Evaluation."""
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
# Shared folders from the repo structure (../datasets, ../models); fall back to local
DATA_DIR = Path("../datasets") if Path("../datasets").exists() else Path("datasets")
MODEL_DIR = Path("../models") if Path("../models").exists() else Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Step 1: Datasets (Feature Matrix Creation)
# - **Placement** and **Loan** datasets: if the CSV is not in `datasets/`, a
#   *synthetic* one is generated so the lab runs anywhere. **Replace
#   `placement.csv` with your own Lab 3 dataset** (keep the target column name
#   or change `target` below).
# - **Disease** dataset: Breast Cancer Wisconsin (built into scikit-learn).

# %% [code] data loading
def make_placement_data(n: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Synthetic student placement data (placed = 1 / 0)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "cgpa": rng.uniform(5.0, 10.0, n).round(2),
        "iq": rng.normal(100, 15, n).round(0),
        "internships": rng.integers(0, 4, n),
        "projects": rng.integers(0, 6, n),
        "communication": rng.integers(1, 11, n),
        "backlogs": rng.integers(0, 5, n),
    })
    score = (1.1 * (df.cgpa - 7) + 0.5 * df.internships + 0.25 * df.projects
             + 0.25 * (df.communication - 5) - 0.7 * df.backlogs
             + 0.02 * (df.iq - 100) + rng.normal(0, 0.8, n))
    df["placed"] = (score > 0.3).astype(int)
    return df


def make_loan_data(n: int = 600, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Synthetic loan data; imbalanced on purpose (fewer approvals)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "income": rng.normal(50000, 18000, n).clip(15000).round(0),
        "loan_amount": rng.normal(150000, 60000, n).clip(20000).round(0),
        "credit_score": rng.normal(650, 80, n).clip(300, 850).round(0),
        "employment_years": rng.integers(0, 20, n),
        "dependents": rng.integers(0, 5, n),
        "existing_loans": rng.integers(0, 4, n),
    })
    score = (0.00004 * df.income - 0.000012 * df.loan_amount
             + 0.012 * (df.credit_score - 650) + 0.08 * df.employment_years
             - 0.25 * df.existing_loans - 0.1 * df.dependents
             + rng.normal(0, 0.6, n))
    df["approved"] = (score > np.percentile(score, 70)).astype(int)  # ~30% approved
    return df


def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for 'placement', 'loan' or 'disease'."""
    if name == "disease":
        data = load_breast_cancer(as_frame=True)
        return data.data, data.target
    makers = {"placement": (make_placement_data, "placed"),
              "loan": (make_loan_data, "approved")}
    if name not in makers:
        raise ValueError(f"Unknown dataset '{name}'. Use: placement, loan, disease")
    maker, target = makers[name]
    path = DATA_DIR / f"{name}.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = maker()
        df.to_csv(path, index=False)
        print(f"   (created synthetic dataset -> {path})")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing in {path}")
    df = df.dropna()                       # simple missing value handling
    return df.drop(columns=[target]), df[target]

# %% [markdown]
# ## Step 2: Train, Predict, Evaluate (reusable pipeline)

# %% [code] train and evaluate
def train_and_evaluate(name: str, X: pd.DataFrame, y: pd.Series,
                       max_depth: int = 4, criterion: str = "gini",
                       show_plots: bool = True) -> Dict:
    """Split, train a Decision Tree, predict and report all metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                   random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_ms = (time.perf_counter() - t0) * 1000
    y_pred = model.predict(X_test)

    metrics = {
        "Dataset": name, "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Train time (ms)": train_ms,
        "CV accuracy (5-fold)": cross_val_score(model, X, y, cv=5).mean(),
    }
    print(f"--- {name} (depth={max_depth}, criterion={criterion}) ---")
    print(classification_report(y_test, y_pred, digits=3))

    if show_plots:
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix: {name}")
        fig.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=120,
                    bbox_inches="tight")
        plt.show()
        plt.close(fig)
    return {"model": model, "metrics": metrics, "X_test": X_test, "y_test": y_test,
            "features": list(X.columns)}

# %% [markdown]
# ## Activity 1: Student Placement Prediction

# %% [code] placement
def demo_placement() -> Dict:
    X, y = load_dataset("placement")
    print("Shape:", X.shape, "| Placed ratio:", round(y.mean(), 2))
    result = train_and_evaluate("placement", X, y, max_depth=4)
    imp = pd.Series(result["model"].feature_importances_, index=result["features"])
    print("Feature importance:\n", imp.sort_values(ascending=False).round(3).to_string())
    joblib.dump({"model": result["model"], "features": result["features"]},
                MODEL_DIR / "decision_tree.pkl")
    print(f"Model saved -> {MODEL_DIR / 'decision_tree.pkl'}")
    return result

# %% [markdown]
# ## Activity 2: Disease Prediction (Breast Cancer, 1 = benign)

# %% [code] disease
def demo_disease() -> Dict:
    X, y = load_dataset("disease")
    print("Shape:", X.shape)
    return train_and_evaluate("disease", X, y, max_depth=4)

# %% [markdown]
# ## Activity 3: Loan Approval Prediction

# %% [code] loan
def demo_loan() -> Dict:
    X, y = load_dataset("loan")
    print("Shape:", X.shape, "| Approved ratio:", round(y.mean(), 2))
    return train_and_evaluate("loan", X, y, max_depth=4)

# %% [markdown]
# ## Activity 4: Decision Tree Visualization
# Graphical tree plus the same tree as readable IF-THEN text rules.

# %% [code] visualization
def demo_visualization() -> None:
    X, y = load_dataset("placement")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
    model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    fig = plt.figure(figsize=(16, 7))
    plot_tree(model, feature_names=list(X.columns), class_names=["Not placed", "Placed"],
              filled=True, rounded=True, fontsize=8)
    fig.savefig(OUTPUT_DIR / "placement_tree.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(export_text(model, feature_names=list(X.columns)))

# %% [markdown]
# ## Application: Intelligent Decision Support System
# Loads the saved model, predicts placement chances for a new student and
# **explains the decision** by listing the tree conditions on the path.

# %% [code] decision support
def explain_prediction(model: DecisionTreeClassifier, features: List[str],
                       sample: Dict[str, float]) -> Tuple[int, float, List[str]]:
    """Return (class, probability, path conditions) for one input record."""
    missing = [f for f in features if f not in sample]
    if missing:
        raise ValueError(f"Missing input values for: {missing}")
    row = pd.DataFrame([sample])[features]
    node_ids = model.decision_path(row).indices
    tree = model.tree_
    steps = []
    for node in node_ids[:-1]:
        feat, thr = features[tree.feature[node]], tree.threshold[node]
        value = row.iloc[0][feat]
        steps.append(f"{feat} = {value} {'<=' if value <= thr else '>'} {thr:.2f}")
    proba = model.predict_proba(row)[0]
    pred = int(model.predict(row)[0])
    return pred, float(proba[pred]), steps


def demo_decision_support() -> None:
    try:
        saved = joblib.load(MODEL_DIR / "decision_tree.pkl")
    except FileNotFoundError:
        print("Model not found - run demo_placement() first.")
        return
    students = {
        "Student A (strong)": dict(cgpa=8.9, iq=115, internships=2, projects=4,
                                   communication=8, backlogs=0),
        "Student B (weak)": dict(cgpa=6.1, iq=95, internships=0, projects=1,
                                 communication=4, backlogs=3),
    }
    for label, record in students.items():
        pred, conf, steps = explain_prediction(saved["model"], saved["features"], record)
        print(f"{label}: {'PLACED' if pred else 'NOT PLACED'} (confidence {conf:.0%})")
        for s in steps:
            print("    because", s)

# %% [markdown]
# ## Analysis 1: Evaluate Decision Tree Performance
# Effect of `max_depth` (overfitting) and of the split criterion (Gini vs Entropy).

# %% [code] depth analysis
def demo_depth_analysis() -> pd.DataFrame:
    X, y = load_dataset("placement")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    rows = []
    for depth in range(1, 13):
        m = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        m.fit(X_train, y_train)
        rows.append({"max_depth": depth,
                     "train_acc": m.score(X_train, y_train),
                     "test_acc": m.score(X_test, y_test)})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    best = df.loc[df.test_acc.idxmax(), "max_depth"]
    print("Best test accuracy at max_depth =", int(best))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.max_depth, df.train_acc, marker="o", label="Train")
    ax.plot(df.max_depth, df.test_acc, marker="s", label="Test")
    ax.set(xlabel="max_depth", ylabel="Accuracy", title="Overfitting analysis")
    ax.legend()
    fig.savefig(OUTPUT_DIR / "depth_vs_accuracy.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("\nGini vs Entropy (max_depth=4):")
    for crit in ("gini", "entropy"):
        r = train_and_evaluate("placement", X, y, 4, crit, show_plots=False)
        print(f"   {crit:<8} accuracy={r['metrics']['Accuracy']:.3f}  "
              f"f1={r['metrics']['F1']:.3f}")
    return df

# %% [markdown]
# ## Analysis 2: Compare Evaluation Metrics
# On the **imbalanced** loan data, accuracy alone can mislead. A model that
# always predicts "rejected" is compared with the Decision Tree.

# %% [code] metric comparison
def demo_metric_comparison() -> pd.DataFrame:
    summary = []
    for name in ("placement", "disease", "loan"):
        X, y = load_dataset(name)
        res = train_and_evaluate(name, X, y, max_depth=4, show_plots=False)
        summary.append(res["metrics"])
    df = pd.DataFrame(summary).round(3)
    print("\n", df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    X, y = load_dataset("loan")
    baseline_acc = 1 - y.mean()
    print(f"\nBaseline 'always reject' accuracy on loan data = {baseline_acc:.2f} "
          f"(but recall for approvals = 0.00)")
    ax = df.set_index("Dataset")[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(
        figsize=(7, 4), rot=0, title="Metric comparison across datasets")
    ax.set_ylim(0, 1.05)
    ax.figure.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(ax.figure)
    return df

# %% [code] main
def main() -> None:
    sections = [("1. Student Placement Prediction", demo_placement),
                ("2. Disease Prediction", demo_disease),
                ("3. Loan Approval Prediction", demo_loan),
                ("4. Decision Tree Visualization", demo_visualization),
                ("5. Decision Support System (Application)", demo_decision_support),
                ("6. Depth / Criterion Analysis", demo_depth_analysis),
                ("7. Metric Comparison", demo_metric_comparison)]
    for title, func in sections:
        print(f"\n===== {title} =====")
        func()


if __name__ == "__main__":
    main()
