# Meter OCR (PaddleOCR) - End User → Server → Admin

## What it does
- User page: upload meter image
- Server: runs PaddleOCR
- Admin page: shows extracted text + uploaded image + raw OCR JSON

## Prerequisites
- Python 3.10+ recommended
- VS Code

## Setup (macOS M1)
1) Open terminal and go to project root.
2) Create virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

3) Install FastAPI + PaddleOCR dependencies:

   pip install -r server/requirements.txt

4) Install PaddlePaddle (Apple Silicon)
   PaddleOCR requires paddlepaddle.
   On macOS M1, you must install an Apple Silicon-compatible paddlepaddle wheel.

   ✅ Recommended approach:
   - Follow PaddlePaddle official install instructions for macOS (ARM64)
   - Then run:
     pip install paddlepaddle

5) Run the server:

   uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

## URLs
- User upload page: http://localhost:8000/
- Admin page: http://localhost:8000/admin

## Notes
- Uploads stored in: server/uploads/
- SQLite DB stored in: server/data/meter_ocr.db
