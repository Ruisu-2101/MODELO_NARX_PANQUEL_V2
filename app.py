import os
import re
import shutil
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import traceback
from torch import nn

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from fastapi import HTTPException

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = None  # permite correr sin BD si no está seteada

engine = None
SessionLocal = None

if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=3,
        max_overflow=2
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def load_dataset_from_db(table_name: str = "pedido") -> pd.DataFrame:
    db = SessionLocal()
    try:
        result = db.execute(text(f'SELECT * FROM "{table_name}"'))
        rows = result.fetchall()
        cols = list(result.keys())
        df = pd.DataFrame(rows, columns=cols)

        # Detectar columnas meta
        col_producto = "producto_nombre" if "producto_nombre" in df.columns else None
        col_prov = "proveedor_id" if "proveedor_id" in df.columns else None

        if not col_producto or not col_prov:
            raise HTTPException(
                status_code=400,
                detail=f"No encontré columnas producto/proveedor. Columnas: {df.columns.tolist()}"
            )

        # Columnas a excluir (no son semanas)
        exclude = {"id", "cantidad", col_producto, col_prov}

        # Semanas = todo lo demás
        week_cols = [c for c in df.columns if c not in exclude]

        # Convertir semanas a numérico
        df[week_cols] = df[week_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Orden final y nombres estándar (para reutilizar tu pipeline)
        df = df[[col_producto, col_prov] + week_cols].copy()
        df = df.rename(columns={col_producto: "Producto", col_prov: "Provedores"})

        return df
    finally:
        db.close()

# =========================
# CONFIG GENERAL
# =========================
APP_DATA_DIR = os.getenv("APP_DATA_DIR", "/tmp/uploads")
os.makedirs(APP_DATA_DIR, exist_ok=True)

LAGS = 6
DEFAULT_EPOCHS = int(os.getenv("EPOCHS", "100"))  # default bajo para Render
LR = float(os.getenv("LR", "0.01"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
SEED = int(os.getenv("SEED", "42"))

np.random.seed(SEED)
torch.manual_seed(SEED)

# Columnas esperadas en dataset "mejorado"
META_COLS = ["ID", "Producto", "Unidad", "Provedores"]

# Fallbacks comunes
FALLBACK_PRODUCT_COLS = ["Producto", "Unnamed: 0", "PRODUCTO", "product"]
FALLBACK_PROVIDER_COLS = ["Provedores", "Proveedores", "Proveedor", "PROVEEDOR", "provider"]

app = FastAPI(title="Panquel Predict API")


# =========================
# UTILIDADES DE ARCHIVOS
# =========================
def latest_uploaded_file() -> str:
    files = [f for f in os.listdir(APP_DATA_DIR) if f.lower().endswith(".csv")]
    if not files:
        raise HTTPException(status_code=404, detail="No hay CSV subido. Usa /upload.")
    files.sort()  # timestamp al inicio => orden correcto
    return os.path.join(APP_DATA_DIR, files[-1])


# =========================
# UTILIDADES DE DATASET
# =========================
def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise HTTPException(status_code=400, detail=f"No encontré ninguna columna válida entre: {candidates}")


def split_groups(cell: Any) -> List[str]:
    """Separa 'CON, SOR / ALA|X' -> ['CON','SOR','ALA','X']"""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    parts = re.split(r"[,\;/\|\+]+", s)
    return [p.strip() for p in parts if p.strip()]


def get_week_cols(df: pd.DataFrame) -> List[str]:
    """Columnas numéricas de semanas: todo lo que NO sea meta conocida."""
    # Si el dataset trae meta cols distintas, igual funcionará porque filtramos con fallback.
    known = set(META_COLS)
    # También excluimos columnas de proveedor/producto si tienen otro nombre
    for c in FALLBACK_PRODUCT_COLS + FALLBACK_PROVIDER_COLS:
        known.add(c)
    week_cols = [c for c in df.columns if c not in known]
    # Filtra columnas que realmente tengan valores numéricos
    if not week_cols:
        # Último recurso: tomar todas menos la 1ra si parece ser nombre
        if len(df.columns) > 1:
            week_cols = df.columns[1:].tolist()
    return week_cols


def list_groups(df: pd.DataFrame, provider_col: str) -> List[str]:
    groups = set()
    for v in df[provider_col].values:
        for g in split_groups(v):
            groups.add(g)
    return sorted(groups)


# =========================
# VENTANAS + MODELO
# =========================
def make_windows_narx(data_matrix: np.ndarray, exo_series: np.ndarray, lags: int = 6):
    X_list, y_list = [], []
    n_products, n_weeks = data_matrix.shape
    for p in range(n_products):
        y_series = data_matrix[p]
        for t in range(lags, n_weeks):
            y_lags = y_series[t - lags : t]
            x_lags = exo_series[t - lags : t]
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


def train_and_predict(
    df: pd.DataFrame,
    group: str,
    epochs: int,
) -> pd.DataFrame:
    # Detectar columnas
    product_col = pick_first_existing(df, FALLBACK_PRODUCT_COLS)
    provider_col = pick_first_existing(df, FALLBACK_PROVIDER_COLS)
    week_cols = get_week_cols(df)

    # Matriz numérica
    weeks_df = df[week_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    data_all = weeks_df.values.astype(np.float32)

    # Filtrar por grupo si aplica
    if group.upper() == "ALL":
        df_group = df.copy()
        data_group = data_all
    else:
        mask = df[provider_col].apply(lambda x: group in split_groups(x))
        df_group = df.loc[mask].copy()
        data_group = data_all[mask.values]
        if df_group.empty:
            raise HTTPException(status_code=404, detail=f"No hay productos para el grupo: {group}")

    # Exógena global (recomendado)
    exo_total = data_all.sum(axis=0).astype(np.float32)

    # Ventanas
    X, y = make_windows_narx(data_group, exo_total, lags=LAGS)
    if len(X) < 10:
        raise HTTPException(status_code=400, detail="Dataset insuficiente para entrenar con LAGS=6.")

    # Split temporal
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)

    # Modelo
    model = NeuralNARX(input_size=2 * LAGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # Entrenamiento
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    # Predicción próxima semana por producto
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

    # Armar resultados
    out = pd.DataFrame(
        {
            "Producto": df_group[product_col].astype(str).values,
            "Provedores": df_group[provider_col].astype(str).values,
            "Pred_Sig_Semana": preds_round,
        }
    ).sort_values("Pred_Sig_Semana", ascending=False).reset_index(drop=True)

    return out


# =========================
# ENDPOINTS
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Panquel Predict - Upload</h2>
    <form action="/upload" enctype="multipart/form-data" method="post">
      <input name="file" type="file" />
      <button type="submit">Subir</button>
    </form>
    <p>Archivos: <a href="/files">/files</a></p>
    <p>Grupos: <a href="/groups">/groups</a></p>
    <p>Docs: <a href="/docs">/docs</a></p>
    """


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Solo se permite .csv")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = file.filename.replace(" ", "_")
    save_path = os.path.join(APP_DATA_DIR, f"{ts}_{safe}")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"ok": True, "saved_as": os.path.basename(save_path), "path": APP_DATA_DIR}


@app.get("/files")
def list_files():
    files = sorted([f for f in os.listdir(APP_DATA_DIR) if f.lower().endswith(".csv")])
    return {"path": APP_DATA_DIR, "files": files, "latest": os.path.basename(latest_uploaded_file()) if files else None}


@app.get("/groups")
def api_groups():
    path = latest_uploaded_file()
    df = pd.read_csv(path)

    provider_col = pick_first_existing(df, FALLBACK_PROVIDER_COLS)
    groups = list_groups(df, provider_col)

    return {"latest_file": os.path.basename(path), "provider_col": provider_col, "groups": groups}


@app.get("/predict")
def api_predict(
    group: str = Query("ALL", description='Grupo a predecir. "ALL" para todos. Ej: SOR'),
    epochs: int = Query(DEFAULT_EPOCHS, ge=5, le=600, description="Épocas (más = más lento)"),
    top: int = Query(20, ge=0, le=50000, description="0 = todos; si no, Top N"),
    group_by_provider: bool = Query(False, description="Si true y group=ALL, agrupa por Provedores"),
):
    path = latest_uploaded_file()
    df = pd.read_csv(path)

    results = train_and_predict(df, group=group, epochs=epochs)
    total = int(results["Pred_Sig_Semana"].sum())

    # Top / todos
    out_df = results if top == 0 else results.head(top)

    # Agrupado por proveedor (solo cuando group=ALL)
    if group.upper() == "ALL" and group_by_provider:
        sorted_df = results.sort_values(["Provedores", "Pred_Sig_Semana"], ascending=[True, False])
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for prov, sub in sorted_df.groupby("Provedores", sort=True):
            grouped[str(prov)] = sub.to_dict(orient="records")

        return {
            "latest_file": os.path.basename(path),
            "group": group,
            "epochs": epochs,
            "lags": LAGS,
            "grouped_by_provider": grouped,
            "total_pred_next_week": total,
        }

    return {
        "latest_file": os.path.basename(path),
        "group": group,
        "epochs": epochs,
        "lags": LAGS,
        "results": out_df.to_dict(orient="records"),
        "total_pred_next_week": total,
    }

@app.get("/db-test")
def db_test():
    db = SessionLocal()
    try:
        row = db.execute(text("select now() as now;")).mappings().first()
        return {"ok": True, "now": str(row["now"])}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        db.close()

@app.get("/db-pedido-info")
def db_pedido_info(limit: int = 3):
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT * FROM pedido LIMIT :n"), {"n": limit})
        rows = result.fetchall()
        cols = list(result.keys())

        preview = []
        for r in rows:
            preview.append({cols[i]: (str(r[i]) if r[i] is not None else None) for i in range(len(cols))})

        return {"ok": True, "columns": cols, "preview": preview}
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}
    finally:
        db.close()

@app.get("/predict-db")
def api_predict_db(
    group: str = Query("ALL", description='Grupo a predecir. "ALL" para todos. Ej: SOR'),
    epochs: int = Query(DEFAULT_EPOCHS, ge=5, le=600, description="Épocas (más = más lento)"),
    top: int = Query(20, ge=0, le=50000, description="0 = todos; si no, Top N"),
    group_by_provider: bool = Query(False, description="Si true y group=ALL, agrupa por Provedores"),
):
    # 1) Cargar tabla pedido -> DataFrame tipo CSV
    df = load_dataset_from_db("pedido")

    # 2) Reusar tu pipeline actual (NARX)
    results = train_and_predict(df, group=group, epochs=epochs)
    total = int(results["Pred_Sig_Semana"].sum())

    out_df = results if top == 0 else results.head(top)

    # 3) Si se pide agrupado por proveedor (solo en ALL)
    if group.upper() == "ALL" and group_by_provider:
        sorted_df = results.sort_values(["Provedores", "Pred_Sig_Semana"], ascending=[True, False])
        grouped = {str(k): v.to_dict(orient="records") for k, v in sorted_df.groupby("Provedores", sort=True)}
        return {
            "source": "supabase_table:pedido",
            "group": group,
            "epochs": epochs,
            "lags": LAGS,
            "grouped_by_provider": grouped,
            "total_pred_next_week": total,
        }

    return {
        "source": "supabase_table:pedido",
        "group": group,
        "epochs": epochs,
        "lags": LAGS,
        "results": out_df.to_dict(orient="records"),
        "total_pred_next_week": total,
    }

@app.get("/db/providers")
def db_providers():
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            select distinct proveedor_id
            from pedido
            where proveedor_id is not null and proveedor_id <> ''
            order by proveedor_id;
        """)).fetchall()
        return {"providers": [r[0] for r in rows]}
    finally:
        db.close()