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