from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import os, shutil
from datetime import datetime

APP_DATA_DIR = os.getenv("APP_DATA_DIR", "/tmp/uploads")
os.makedirs(APP_DATA_DIR, exist_ok=True)

app = FastAPI(title="Panquel Predict - Upload")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Panquel Predict - Subir Dataset</h2>
    <form action="/upload" enctype="multipart/form-data" method="post">
      <input name="file" type="file" />
      <button type="submit">Subir</button>
    </form>
    <p>Ver archivos: <a href="/files">/files</a></p>
    """

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".xlsx")):
        raise HTTPException(status_code=400, detail="Solo se permiten .csv o .xlsx")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = file.filename.replace(" ", "_")
    path = os.path.join(APP_DATA_DIR, f"{ts}_{safe}")

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"ok": True, "saved_as": os.path.basename(path)}

@app.get("/files")
def list_files():
    return {"path": APP_DATA_DIR, "files": sorted(os.listdir(APP_DATA_DIR))}

from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from fastapi import Query

# ==========
# Config IA
# ==========
LAGS = 6
DEFAULT_EPOCHS = int(os.getenv("EPOCHS", "100"))  # puedes subirlo luego
LR = float(os.getenv("LR", "0.01"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
SEED = int(os.getenv("SEED", "42"))

np.random.seed(SEED)
torch.manual_seed(SEED)

META_COLS = ["ID", "Producto", "Unidad", "Provedores"]
PRODUCT_COL = "Producto"
GROUP_COL = "Provedores"


def latest_uploaded_file() -> str:
    """Regresa el path del archivo más reciente en APP_DATA_DIR."""
    files = [f for f in os.listdir(APP_DATA_DIR) if f.lower().endswith(".csv")]
    if not files:
        raise HTTPException(status_code=404, detail="No hay CSV subido aún. Sube uno en /upload.")
    files.sort()  # como guardamos con timestamp YYYYMMDD_HHMMSS, sort sirve
    return os.path.join(APP_DATA_DIR, files[-1])


def _split_groups(cell: str):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    parts = re.split(r"[,\;/\|\+]+", s)
    return [p.strip() for p in parts if p.strip()]


def get_groups(df: pd.DataFrame):
    groups = set()
    for v in df[GROUP_COL].values:
        for g in _split_groups(v):
            groups.add(g)
    return sorted(groups)


def make_windows_narx(data_matrix: np.ndarray, exo_series: np.ndarray, lags: int = 6):
    X_list, y_list = [], []
    n_products, n_weeks = data_matrix.shape
    for p in range(n_products):
        y_series = data_matrix[p]
        for t in range(lags, n_weeks):
            y_lags = y_series[t-lags:t]
            x_lags = exo_series[t-lags:t]
            X_list.append(np.concatenate([y_lags, x_lags]))
            y_list.append(y_series[t])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


class NeuralNARX(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_and_predict(df: pd.DataFrame, group: str = "ALL", epochs: int = DEFAULT_EPOCHS):
    # 1) columnas de semanas
    week_cols = [c for c in df.columns if c not in META_COLS]
    if not week_cols:
        raise HTTPException(status_code=400, detail="No detecté columnas de semanas en el CSV.")

    # 2) matriz numérica
    weeks_df = df[week_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    data_all = weeks_df.values.astype(np.float32)

    # 3) filtrar por grupo (si aplica)
    if group.upper() == "ALL":
        df_group = df.copy()
        data_group = data_all
    else:
        mask = df[GROUP_COL].apply(lambda x: group in _split_groups(x))
        df_group = df.loc[mask].copy()
        data_group = data_all[mask.values]
        if df_group.empty:
            raise HTTPException(status_code=404, detail=f"No hay productos para el grupo: {group}")

    # 4) exógena global (recomendado)
    exo_total = data_all.sum(axis=0).astype(np.float32)

    # 5) ventanas NARX
    X, y = make_windows_narx(data_group, exo_total, lags=LAGS)

    # split temporal
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test).unsqueeze(1)

    # 6) modelo
    model = NeuralNARX(input_size=2 * LAGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # 7) entrenamiento rápido
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    # 8) predicción siguiente semana para productos filtrados
    model.eval()
    last_x = exo_total[-LAGS:]
    preds = []
    with torch.no_grad():
        for i in range(data_group.shape[0]):
            last_y = data_group[i, -LAGS:]
            x_in = np.concatenate([last_y, last_x]).astype(np.float32)
            p = model(torch.tensor(x_in).unsqueeze(0)).item()
            preds.append(p)

    preds = np.array(preds)
    preds_round = np.clip(np.rint(preds), 0, None).astype(int)

    results = pd.DataFrame({
        "Producto": df_group[PRODUCT_COL].astype(str).values,
        "Provedores": df_group[GROUP_COL].astype(str).values,
        "Pred_Sig_Semana": preds_round
    }).sort_values("Pred_Sig_Semana", ascending=False).reset_index(drop=True)

    total = int(results["Pred_Sig_Semana"].sum())

    # métricas rápidas (opcional)
    with torch.no_grad():
        y_pred_test = model(X_test_t).squeeze().cpu().numpy() if len(X_test) else np.array([])
        y_true_test = y_test_t.squeeze().cpu().numpy() if len(X_test) else np.array([])

    return results, total, {
        "epochs": epochs,
        "n_products_group": int(data_group.shape[0]),
        "n_weeks": int(data_all.shape[1]),
        "samples": int(len(X)),
        "test_samples": int(len(X_test)),
    }


@app.get("/groups")
def api_groups():
    """
    Lista los grupos disponibles según la columna Provedores.
    """
    path = latest_uploaded_file()
    df = pd.read_csv(path)
    return {"latest_file": os.path.basename(path), "groups": get_groups(df)}


@app.get("/predict")
def api_predict(
    group: str = Query("ALL", description='Grupo a predecir. Usa "ALL" para todos. Ej: CON'),
    epochs: int = Query(DEFAULT_EPOCHS, ge=10, le=600, description="Épocas de entrenamiento (más = más lento)"),
    top: int = Query(20, ge=1, le=200, description="Cuántos resultados regresar")
):
    """
    Entrena un NARX rápido con el último CSV subido y regresa predicción de la siguiente semana.
    """
    path = latest_uploaded_file()
    df = pd.read_csv(path)

    results, total, info = train_and_predict(df, group=group, epochs=epochs)

    return {
        "latest_file": os.path.basename(path),
        "group": group,
        "info": info,
        "top": results.head(top).to_dict(orient="records"),
        "total_pred_next_week": total
    }