# %% [markdown]
# # Lab 6: Deep Learning Fundamentals using PyTorch
# **Course:** IC-709D | AI Lab  **Unit 6 (CO4, CO5)**
#
# **Basics:** Tensor Operations, Automatic Differentiation, Linear Regression,
# Neural Network Basics, Model Training, Loss Functions, Optimizers.
# **Activities:** Tensor Manipulation, Linear Regression, Training Loop,
# Gradient Computation, Model Evaluation.
# **Applications:** House Price Prediction, Student Score Prediction, Sales Forecasting.
# **Analysis:** Traditional ML vs Deep Learning, convergence of gradient-based learning.

# %% [code] imports
"""Lab 6 - Deep Learning Fundamentals using PyTorch."""
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
# Shared folders from the repo structure (../datasets, ../models); fall back to local
DATA_DIR = Path("../datasets") if Path("../datasets").exists() else Path("datasets")
MODEL_DIR = Path("../models") if Path("../models").exists() else Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Activity 1: Tensor Manipulation
# Creation, shapes, indexing, reshaping, arithmetic, broadcasting, matrix
# multiplication and NumPy interoperability.

# %% [code] tensors
def demo_tensors() -> None:
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.ones(2, 3)
    print("a =\n", a)
    print("shape:", tuple(a.shape), "| dtype:", a.dtype, "| device:", a.device)
    print("a + b =\n", a + b)
    print("a * b (element-wise) =\n", a * b)
    print("a @ a.T (matrix multiplication) =\n", a @ a.T)
    print("a.reshape(3, 2) =\n", a.reshape(3, 2))
    print("a[:, 1] (second column):", a[:, 1])
    print("sum / mean / max:", a.sum().item(), a.mean().item(), a.max().item())
    print("broadcasting a + [10, 20, 30] =\n", a + torch.tensor([10.0, 20.0, 30.0]))
    print("arange:", torch.arange(0, 10, 2), "| linspace:", torch.linspace(0, 1, 5))
    print("zeros:\n", torch.zeros(2, 2), "\nrand:\n", torch.rand(2, 2))
    as_numpy = a.numpy()                       # tensor -> NumPy (shares memory)
    back = torch.from_numpy(as_numpy)          # NumPy -> tensor
    print("NumPy round trip equal:", torch.equal(a, back))
    use_cuda = torch.cuda.is_available()
    print("GPU available:", use_cuda)

# %% [markdown]
# ## Activity 2: Gradient Computation (Automatic Differentiation)
# `backward()` computes gradients automatically; we verify them against
# hand-derived (analytic) gradients.

# %% [code] gradients
def demo_gradients() -> None:
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2 + 3 * x                         # dy/dx = 2x + 3
    y.backward()
    print(f"y = x^2 + 3x at x=2: y = {y.item()}, dy/dx = {x.grad.item()} (analytic: 7)")

    w = torch.tensor([1.0, 2.0], requires_grad=True)
    features = torch.tensor([3.0, 4.0])
    loss = ((w * features).sum() - 10.0) ** 2   # L = (w.x - 10)^2
    loss.backward()
    analytic = 2 * ((w.detach() * features).sum() - 10.0) * features
    print("autograd gradient:", w.grad.tolist(), "| analytic gradient:", analytic.tolist())
    print("Gradients match:", torch.allclose(w.grad, analytic))

# %% [markdown]
# ## Activity 3: Linear Regression from Scratch (autograd + manual update)
# Fit `y = 2x + 1` from noisy data using gradient descent written by hand,
# so every step of the training loop is visible.

# %% [code] linear regression scratch
def linear_regression_from_scratch(epochs: int = 500, lr: float = 0.1) -> List[float]:
    torch.manual_seed(SEED)
    x = torch.rand(100, 1)
    y = 2 * x + 1 + 0.1 * torch.randn(100, 1)
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    losses = []
    for epoch in range(epochs):
        pred = x * w + b                        # forward pass
        loss = ((pred - y) ** 2).mean()         # MSE loss
        loss.backward()                         # compute gradients
        with torch.no_grad():                   # gradient descent update
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        losses.append(loss.item())
    print(f"Learned w = {w.item():.3f} (true 2.0), b = {b.item():.3f} (true 1.0)")
    print(f"Final MSE = {losses[-1]:.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(x.numpy(), y.numpy(), s=12, label="data")
    xs = torch.linspace(0, 1, 50).unsqueeze(1)
    axes[0].plot(xs.numpy(), (xs * w.detach() + b.detach()).numpy(), "r", label="fit")
    axes[0].set(title="Linear regression fit", xlabel="x", ylabel="y")
    axes[0].legend()
    axes[1].plot(losses)
    axes[1].set(title="Training loss", xlabel="epoch", ylabel="MSE")
    fig.savefig(OUTPUT_DIR / "linear_regression_scratch.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return losses

# %% [markdown]
# ## Activity 4: Training Loop with `nn.Module`, Loss Functions and Optimizers
# Reusable pipeline: scaling, train/test split, mini-batch training loop with
# `nn.Linear` (linear regression) or `nn.Sequential` (neural network), and
# evaluation with MSE / RMSE / MAE / R2.

# %% [code] training pipeline
def build_model(n_features: int, hidden: Optional[List[int]] = None) -> nn.Module:
    """Linear regression if `hidden` is empty, otherwise an MLP with ReLU."""
    if not hidden:
        return nn.Linear(n_features, 1)
    layers, prev = [], n_features
    for size in hidden:
        layers += [nn.Linear(prev, size), nn.ReLU()]
        prev = size
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    """Create an optimizer: 'sgd', 'momentum' or 'adam'."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"Unknown optimizer '{name}'. Use: sgd, momentum, adam")


def prepare_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                 chronological: bool = False) -> Dict:
    """Split and standardise features/target (scalers fit on training data only)."""
    if chronological:                           # time series: no shuffling
        cut = int(len(X) * (1 - test_size))
        X_tr, X_te, y_tr, y_te = X[:cut], X[cut:], y[:cut], y[cut:]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                                  random_state=SEED)
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    return {
        "X_train": to_tensor(x_scaler.transform(X_tr)),
        "y_train": to_tensor(y_scaler.transform(y_tr.reshape(-1, 1))),
        "X_test": to_tensor(x_scaler.transform(X_te)),
        "y_test_raw": y_te, "X_train_raw": X_tr, "X_test_raw": X_te, "y_train_raw": y_tr,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"R2": r2_score(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred)}


def train_regressor(data: Dict, hidden: Optional[List[int]] = None,
                    optimizer: str = "adam", lr: float = 0.01, epochs: int = 200,
                    batch_size: int = 64) -> Dict:
    """Train a PyTorch regressor and evaluate it on the test set (original units)."""
    torch.manual_seed(SEED)
    X_train, y_train = data["X_train"], data["y_train"]
    model = build_model(X_train.shape[1], hidden)
    loss_fn = nn.MSELoss()
    opt = make_optimizer(optimizer, model.parameters(), lr)
    history, n = [], len(X_train)

    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()                              # reset gradients
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()                              # backpropagation
            opt.step()                                   # update weights
        model.eval()
        with torch.no_grad():
            epoch_loss = loss_fn(model(X_train), y_train).item()
        history.append(epoch_loss)
        if not np.isfinite(epoch_loss):
            print(f"   training diverged at epoch {epoch} (lr={lr}, {optimizer})")
            break
    train_time = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        scaled_pred = model(data["X_test"]).numpy()
    pred = data["y_scaler"].inverse_transform(scaled_pred).ravel()
    metrics = regression_metrics(data["y_test_raw"], pred) if np.all(np.isfinite(pred)) \
        else {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan")}
    return {"model": model, "history": history, "metrics": metrics,
            "train_time": train_time, "pred": pred}


def save_model(result: Dict, data: Dict, features: List[str], path: Path) -> None:
    """Save weights plus the scaling values needed for prediction."""
    torch.save({"state_dict": result["model"].state_dict(),
                "features": features,
                "x_mean": data["x_scaler"].mean_.tolist(),
                "x_scale": data["x_scaler"].scale_.tolist(),
                "y_mean": float(data["y_scaler"].mean_[0]),
                "y_scale": float(data["y_scaler"].scale_[0])}, path)


def load_and_predict(path: Path, record: Dict[str, float]) -> float:
    """Load a saved linear model and predict for one record."""
    saved = torch.load(path, weights_only=True)
    missing = [f for f in saved["features"] if f not in record]
    if missing:
        raise ValueError(f"Missing input values for: {missing}")
    model = nn.Linear(len(saved["features"]), 1)
    model.load_state_dict(saved["state_dict"])
    x = (np.array([record[f] for f in saved["features"]]) - saved["x_mean"]) / saved["x_scale"]
    with torch.no_grad():
        out = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
    return out * saved["y_scale"] + saved["y_mean"]

# %% [markdown]
# ## Applications: Datasets
# House Price, Student Score and Sales Forecasting. If a CSV is not in
# `datasets/`, a **synthetic** one is generated. **Replace these with real
# datasets for your final submission** (keep the target column name, or edit
# `DATASETS` below).

# %% [code] datasets
def make_house_data(n: int = 500, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"area": rng.uniform(500, 3000, n).round(0),
                       "bedrooms": rng.integers(1, 6, n),
                       "age": rng.integers(0, 41, n),
                       "distance_km": rng.uniform(1, 30, n).round(1)})
    df["price"] = (4000 * df.area + 150000 * df.bedrooms - 8000 * df.age
                   - 12000 * df.distance_km + 0.5 * df.area ** 2
                   + rng.normal(0, 80000, n)).round(0)
    return df


def make_student_data(n: int = 400, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"hours_studied": rng.uniform(0, 10, n).round(1),
                       "attendance": rng.uniform(50, 100, n).round(0),
                       "previous_score": rng.uniform(30, 100, n).round(0)})
    score = (8 * np.sqrt(df.hours_studied) + 0.2 * df.attendance
             + 0.45 * df.previous_score + rng.normal(0, 3, n))
    df["final_score"] = score.clip(0, 100).round(1)
    return df


def make_sales_data(months: int = 120, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(months)
    sales = 200 + 3 * t + 40 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 10, months)
    return pd.DataFrame({"t": t, "month_sin": np.sin(2 * np.pi * t / 12),
                         "month_cos": np.cos(2 * np.pi * t / 12),
                         "sales": sales.round(1)})


DATASETS = {   # name: (generator, target column, chronological split?)
    "house_price": (make_house_data, "price", False),
    "student_score": (make_student_data, "final_score", False),
    "sales": (make_sales_data, "sales", True),
}


def load_regression_data(name: str) -> Dict:
    """Load a dataset (creating a synthetic CSV if missing) and prepare tensors."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Use: {list(DATASETS)}")
    maker, target, chronological = DATASETS[name]
    path = DATA_DIR / f"{name}.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = maker()
        df.to_csv(path, index=False)
        print(f"   (created synthetic dataset -> {path})")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing in {path}")
    df = df.dropna()
    features = [c for c in df.columns if c != target]
    data = prepare_data(df[features].to_numpy(dtype=float),
                        df[target].to_numpy(dtype=float), chronological=chronological)
    data["features"], data["target"] = features, target
    return data

# %% [markdown]
# ## Application 1: House Price Prediction (Linear Regression)

# %% [code] house price
def demo_house_price() -> None:
    data = load_regression_data("house_price")
    result = train_regressor(data, hidden=None, optimizer="adam", lr=0.05, epochs=200)
    print("Test metrics:", {k: round(v, 3) for k, v in result["metrics"].items()})
    save_model(result, data, data["features"], MODEL_DIR / "linear_regression.pt")
    print(f"Model saved -> {MODEL_DIR / 'linear_regression.pt'}")
    sample = {"area": 1800, "bedrooms": 3, "age": 10, "distance_km": 8}
    price = load_and_predict(MODEL_DIR / "linear_regression.pt", sample)
    print(f"Predicted price for {sample}: {price:,.0f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(data["y_test_raw"], result["pred"], s=12)
    lims = [data["y_test_raw"].min(), data["y_test_raw"].max()]
    ax.plot(lims, lims, "r--")
    ax.set(title="House price: actual vs predicted", xlabel="Actual", ylabel="Predicted")
    fig.savefig(OUTPUT_DIR / "house_price_pred_vs_actual.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## Application 2: Student Score Prediction (Neural Network)

# %% [code] student score
def demo_student_score() -> None:
    data = load_regression_data("student_score")
    result = train_regressor(data, hidden=[16, 8], optimizer="adam", lr=0.01, epochs=200)
    print("Test metrics:", {k: round(v, 3) for k, v in result["metrics"].items()})
    print("Final training loss (standardised MSE):", round(result["history"][-1], 4))

# %% [markdown]
# ## Application 3: Sales Forecasting (chronological split)
# The last 20% of months are held out as the "future" to forecast.

# %% [code] sales forecasting
def demo_sales_forecast() -> None:
    data = load_regression_data("sales")
    result = train_regressor(data, hidden=None, optimizer="adam", lr=0.05, epochs=300)
    print("Forecast metrics:", {k: round(v, 3) for k, v in result["metrics"].items()})
    n_train = len(data["y_train_raw"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(n_train), data["y_train_raw"], label="train (actual)")
    ax.plot(range(n_train, n_train + len(result["pred"])), data["y_test_raw"],
            label="test (actual)")
    ax.plot(range(n_train, n_train + len(result["pred"])), result["pred"], "r--",
            label="forecast")
    ax.set(title="Sales forecasting", xlabel="month", ylabel="sales")
    ax.legend()
    fig.savefig(OUTPUT_DIR / "sales_forecast.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## Analysis 1: Traditional ML vs Deep Learning
# Scikit-learn Linear Regression and Random Forest vs PyTorch linear model and
# a small neural network, on all three datasets.

# %% [code] ml vs dl
def demo_ml_vs_dl() -> pd.DataFrame:
    rows = []
    for name in DATASETS:
        data = load_regression_data(name)
        X_tr, X_te = data["X_train_raw"], data["X_test_raw"]
        y_tr, y_te = data["y_train_raw"], data["y_test_raw"]
        for label, model in [("Sklearn LinearRegression", LinearRegression()),
                             ("Sklearn RandomForest", RandomForestRegressor(
                                 n_estimators=100, random_state=SEED))]:
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            elapsed = time.perf_counter() - t0
            rows.append({"Dataset": name, "Model": label, "Train time (s)": elapsed,
                         **regression_metrics(y_te, model.predict(X_te))})
        for label, hidden in [("PyTorch Linear", None), ("PyTorch MLP [32,16]", [32, 16])]:
            res = train_regressor(data, hidden=hidden, optimizer="adam",
                                  lr=0.01, epochs=200)
            rows.append({"Dataset": name, "Model": label,
                         "Train time (s)": res["train_time"], **res["metrics"]})
    df = pd.DataFrame(rows).round(4)
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "ml_vs_dl_comparison.csv", index=False)
    return df

# %% [markdown]
# ## Analysis 2: Convergence of Gradient-Based Learning
# Effect of the optimizer and learning rate on the training loss curve
# (house price data). A too-large learning rate makes SGD diverge.

# %% [code] convergence
def demo_convergence() -> pd.DataFrame:
    data = load_regression_data("house_price")
    configs = [("SGD lr=0.01", "sgd", 0.01), ("SGD lr=0.1", "sgd", 0.1),
               ("SGD lr=1.5 (too large)", "sgd", 1.5),
               ("Momentum lr=0.01", "momentum", 0.01), ("Adam lr=0.01", "adam", 0.01)]
    histories, rows = {}, []
    for label, opt_name, lr in configs:
        res = train_regressor(data, hidden=[16], optimizer=opt_name, lr=lr, epochs=100)
        histories[label] = res["history"]
        rows.append({"Config": label, "Final train loss": res["history"][-1],
                     "Test R2": res["metrics"]["R2"], "Epochs run": len(res["history"])})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, hist in histories.items():
        ax.plot(hist, label=label)
    ax.set(yscale="log", xlabel="epoch", ylabel="training loss (standardised MSE)",
           title="Convergence of optimizers / learning rates")
    ax.legend()
    fig.savefig(OUTPUT_DIR / "convergence_comparison.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    df.to_csv(OUTPUT_DIR / "convergence_summary.csv", index=False)
    return df

# %% [code] main
def main() -> None:
    sections = [("1. Tensor Manipulation", demo_tensors),
                ("2. Gradient Computation", demo_gradients),
                ("3. Linear Regression from Scratch",
                 lambda: linear_regression_from_scratch()),
                ("4. House Price Prediction", demo_house_price),
                ("5. Student Score Prediction", demo_student_score),
                ("6. Sales Forecasting", demo_sales_forecast),
                ("7. Traditional ML vs Deep Learning", demo_ml_vs_dl),
                ("8. Convergence Analysis", demo_convergence)]
    for title, func in sections:
        print(f"\n===== {title} =====")
        func()


if __name__ == "__main__":
    main()
