# Meter OCR - End User -> Server -> Admin

## What it does
- User page: upload meter image
- Server: runs OCR on uploaded meter images
- Admin page: shows extracted text + uploaded image + raw OCR JSON

## Prerequisites
- Python 3.10+ recommended
- VS Code

## Setup (macOS M1)
1) Open terminal and go to project root.
2) Create virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

3) Install backend dependencies:

   pip install -r server/requirements.txt

4) Run the server:

   uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

## Google Cloud Vision OCR (for uploaded images)
- Install deps: `pip install -r server/requirements.txt`
- Configure Google Vision API key:
  - `export GCV_API_KEY=...`
- OCR backend mode:
  - `export OCR_BACKEND=gcv_then_tesseract` (default, GCV first then local fallback)
  - `export OCR_BACKEND=gcv` (GCV only)
  - `export OCR_BACKEND=tesseract` (local only)

## URLs
- User upload page: http://localhost:8000/
- Admin page: http://localhost:8000/admin

## Notes
- Uploads stored in: server/uploads/
- SQLite DB stored in: server/data/meter_ocr.db
