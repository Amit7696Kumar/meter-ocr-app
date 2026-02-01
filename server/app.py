import os
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, Form, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.concurrency import run_in_threadpool

from passlib.hash import bcrypt

from server.db import (
    init_db,
    create_user,
    get_user_by_username,
    get_user_by_id,
    insert_reading,
    fetch_readings_all,
    fetch_readings_by_team,
    fetch_readings_by_user,
    create_alert,
    fetch_alerts_for_admin,
    fetch_alerts_for_coadmin,
    mark_alert_read,
    count_unread_admin,
    count_unread_coadmin,
)

# ✅ use warmup + run_ocr from ocr_engine
from server.ocr_engine import run_ocr, warmup_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)

IDEAL_PATH = os.path.join(BASE_DIR, "ideal_values.json")

DEBUG_DIR = Path(os.path.join(BASE_DIR, "static", "debug"))
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Meter OCR (YOLO + EasyOCR)")

# Session cookies
app.add_middleware(SessionMiddleware, secret_key="CHANGE_ME_TO_A_RANDOM_LONG_SECRET")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=TEMPLATE_DIR)


# -----------------------
# Ideals + helper utils
# -----------------------
def load_ideals():
    if not os.path.exists(IDEAL_PATH):
        print(f"[IDEALS] ⚠️ Missing {IDEAL_PATH}. Using no rules.", flush=True)
        return {}

    try:
        with open(IDEAL_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                print(f"[IDEALS] ⚠️ {IDEAL_PATH} is empty. Using no rules.", flush=True)
                return {}
            data = json.loads(raw)
            if not isinstance(data, dict):
                print(f"[IDEALS] ⚠️ {IDEAL_PATH} is not a JSON object. Using no rules.", flush=True)
                return {}
            return data
    except Exception as e:
        print(f"[IDEALS] ❌ Failed to read/parse {IDEAL_PATH}: {e}", flush=True)
        return {}


def current_user(request: Request):
    uid = request.session.get("uid")
    if not uid:
        return None
    return get_user_by_id(int(uid))


def require_role(user, roles):
    return user and user.get("role") in roles


def compare_against_ideal(meter_type: str, value_str: Optional[str]):
    ideals = load_ideals()
    rule = ideals.get(meter_type)
    if not rule:
        return (False, None, None)

    try:
        v = float(value_str)
    except Exception:
        return (False, None, None)

    mn = rule.get("min")
    mx = rule.get("max")

    if mn is not None and v < float(mn):
        return (True, "low", f"{meter_type.upper()} is LOW: {v} (min ideal {mn})")
    if mx is not None and v > float(mx):
        return (True, "high", f"{meter_type.upper()} is HIGH: {v} (max ideal {mx})")

    return (False, None, None)


# -----------------------
# Startup
# -----------------------
import threading

@app.on_event("startup")
def startup():
    init_db()

    # ✅ Start server immediately, warm up in background
    def _warm():
        try:
            print("[WARMUP] Starting in background...", flush=True)
            warmup_models()
            print("[WARMUP] Done.", flush=True)
        except Exception as e:
            print(f"[WARMUP] ⚠️ Failed: {e}", flush=True)

    threading.Thread(target=_warm, daemon=True).start()

    # create users...
    if not get_user_by_username("admin"):
        create_user("admin", bcrypt.hash("admin123"), "admin", None)

    for t in range(1, 6):
        uname = f"coadmin{t}"
        if not get_user_by_username(uname):
            create_user(uname, bcrypt.hash("coadmin123"), "coadmin", t)

    for t in range(1, 6):
        for i in range(1, 9):
            uname = f"user{t}{i}"
            if not get_user_by_username(uname):
                create_user(uname, bcrypt.hash("user123"), "user", t)

    print("[STARTUP] App ready. Visit http://127.0.0.1:8000/login", flush=True)


# -----------------------
# Auth
# -----------------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def do_login(request: Request, username: str = Form(...), password: str = Form(...)):
    u = get_user_by_username(username.strip())
    if not u or not bcrypt.verify(password, u["password_hash"]):
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": "Invalid credentials"}, status_code=401
        )

    request.session["uid"] = int(u["id"])
    role = u["role"]

    if role == "admin":
        return RedirectResponse("/admin", status_code=303)
    if role == "coadmin":
        return RedirectResponse(f"/coadmin/{u['team']}", status_code=303)
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# -----------------------
# User page + Upload
# -----------------------
@app.get("/", response_class=HTMLResponse)
def user_page(request: Request):
    u = current_user(request)
    if not require_role(u, ["user", "coadmin", "admin"]):
        return RedirectResponse("/login", status_code=303)

    if u["role"] == "admin":
        return RedirectResponse("/admin", status_code=303)
    if u["role"] == "coadmin":
        return RedirectResponse(f"/coadmin/{u['team']}", status_code=303)

    my_readings = fetch_readings_by_user(int(u["id"]))
    return templates.TemplateResponse(
        "user.html",
        {"request": request, "user": u, "readings": my_readings},
    )


@app.post("/upload")
async def upload_meter_image(
    request: Request,
    label: str = Form(...),
    meter_type: str = Form(...),
    image: UploadFile = File(...),
):
    u = current_user(request)
    if not require_role(u, ["user"]):
        return RedirectResponse("/login", status_code=303)

    if meter_type not in ["earthing", "temp"]:
        return templates.TemplateResponse(
            "user.html",
            {
                "request": request,
                "user": u,
                "readings": fetch_readings_by_user(int(u["id"])),
                "error": "Invalid meter type",
            },
            status_code=400,
        )

    if not image.filename:
        return templates.TemplateResponse(
            "user.html",
            {
                "request": request,
                "user": u,
                "readings": fetch_readings_by_user(int(u["id"])),
                "error": "Please select an image.",
            },
            status_code=400,
        )

    ext = os.path.splitext(image.filename.lower())[1]
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        return templates.TemplateResponse(
            "user.html",
            {
                "request": request,
                "user": u,
                "readings": fetch_readings_by_user(int(u["id"])),
                "error": "Only image files are allowed.",
            },
            status_code=400,
        )

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    # Save upload
    with open(filepath, "wb") as f:
        f.write(await image.read())

    # ✅ OCR in threadpool so request doesn't feel "stuck"
    debug_id = uuid.uuid4().hex
    try:
        ocr_result = await run_in_threadpool(run_ocr, filepath, debug_id)
    except Exception as e:
        print(f"[OCR] ❌ Failed: {e}", flush=True)
        return templates.TemplateResponse(
            "user.html",
            {
                "request": request,
                "user": u,
                "readings": fetch_readings_by_user(int(u["id"])),
                "error": f"OCR failed: {e}",
            },
            status_code=500,
        )

    numeric_value: Optional[str] = None
    if ocr_result.get("numeric"):
        numeric_value = ocr_result["numeric"].get("value")

    # Store reading
    rid = insert_reading(
        user_id=int(u["id"]),
        team=int(u["team"]),
        meter_type=meter_type,
        label=label.strip(),
        value=numeric_value,
        filename=filename,
        ocr_json=json.dumps(ocr_result, default=str),
    )

    # Alerts logic
    print(f"[ALERT] meter_type={meter_type} numeric_value={numeric_value}", flush=True)
    is_alert, severity, msg = compare_against_ideal(meter_type, numeric_value)
    print(f"[ALERT] is_alert={is_alert} severity={severity} msg={msg}", flush=True)

    if numeric_value and is_alert:
        # coadmin (team)
        create_alert(
            reading_id=rid,
            target_role="coadmin",
            target_team=int(u["team"]),
            message=f"Team {u['team']} - {label}: {msg}",
            severity=severity,
        )
        # admin
        create_alert(
            reading_id=rid,
            target_role="admin",
            target_team=None,
            message=f"Team {u['team']} - {label}: {msg}",
            severity=severity,
        )

    return RedirectResponse("/success", status_code=303)


@app.get("/success", response_class=HTMLResponse)
def success_page(request: Request):
    u = current_user(request)
    if not u:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("success.html", {"request": request, "user": u})


# -----------------------
# Coadmin pages
# -----------------------
@app.get("/coadmin/{team_id}", response_class=HTMLResponse)
def coadmin_page(request: Request, team_id: int):
    u = current_user(request)
    if not require_role(u, ["coadmin", "admin"]):
        return RedirectResponse("/login", status_code=303)

    if u["role"] == "coadmin" and int(u["team"]) != int(team_id):
        return HTMLResponse("Forbidden", status_code=403)

    readings = fetch_readings_by_team(int(team_id))
    alerts = fetch_alerts_for_coadmin(int(team_id), unread_only=False)
    unread = count_unread_coadmin(int(team_id))

    return templates.TemplateResponse(
        "coadmin.html",
        {
            "request": request,
            "user": u,
            "team_id": team_id,
            "readings": readings,
            "alerts": alerts,
            "unread_count": unread,
        },
    )


# -----------------------
# Admin pages
# -----------------------
@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    u = current_user(request)
    if not require_role(u, ["admin"]):
        return RedirectResponse("/login", status_code=303)

    readings = fetch_readings_all()
    alerts = fetch_alerts_for_admin(unread_only=False)
    unread = count_unread_admin()

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "user": u,
            "readings": readings,
            "alerts": alerts,
            "unread_count": unread,
            "teams": [1, 2, 3, 4, 5],
        },
    )


# -----------------------
# CSV Downloads
# -----------------------
import csv

def _csv_rows(readings):
    header = ["created_at", "team", "username", "meter_type", "label", "value", "filename"]
    yield header
    for r in readings:
        # r is dict-like
        yield [
            r.get("created_at", ""),
            r.get("team", ""),
            r.get("username", ""),
            r.get("meter_type", ""),
            r.get("label", ""),
            r.get("value", ""),
            r.get("filename", ""),
        ]


def _stream_csv(filename, rows):
    def gen():
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in rows:
            buf.seek(0)
            buf.truncate(0)
            writer.writerow(row)
            yield buf.getvalue()

    return StreamingResponse(
        gen(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/download/csv/my")
def download_my_csv(request: Request):
    u = current_user(request)
    if not require_role(u, ["user"]):
        return RedirectResponse("/login", status_code=303)
    readings = fetch_readings_by_user(int(u["id"]))
    return _stream_csv("my_readings.csv", _csv_rows(readings))


@app.get("/download/csv/team/{team_id}")
def download_team_csv(request: Request, team_id: int):
    u = current_user(request)
    if not require_role(u, ["coadmin", "admin"]):
        return RedirectResponse("/login", status_code=303)
    if u["role"] == "coadmin" and int(u["team"]) != int(team_id):
        return HTMLResponse("Forbidden", status_code=403)
    readings = fetch_readings_by_team(int(team_id))
    return _stream_csv(f"team_{team_id}_readings.csv", _csv_rows(readings))


@app.get("/download/csv/admin")
def download_admin_csv(request: Request):
    u = current_user(request)
    if not require_role(u, ["admin"]):
        return RedirectResponse("/login", status_code=303)
    readings = fetch_readings_all()
    return _stream_csv("all_readings.csv", _csv_rows(readings))


# -----------------------
# Alerts API (live refresh)
# -----------------------
@app.get("/api/alerts/admin")
def api_alerts_admin(request: Request):
    u = current_user(request)
    if not require_role(u, ["admin"]):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    alerts = fetch_alerts_for_admin(unread_only=False)
    unread = count_unread_admin()
    return {"unread_count": unread, "alerts": alerts}


# ✅ your JS likely calls: /api/alerts/coadmin?team=1
@app.get("/api/alerts/coadmin")
def api_alerts_coadmin(request: Request, team: int):
    u = current_user(request)
    if not require_role(u, ["coadmin", "admin"]):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    if u["role"] == "coadmin" and int(u["team"]) != int(team):
        return JSONResponse({"error": "forbidden"}, status_code=403)

    alerts = fetch_alerts_for_coadmin(int(team), unread_only=False)
    unread = count_unread_coadmin(int(team))
    return {"unread_count": unread, "alerts": alerts}


@app.post("/alerts/{alert_id}/read")
async def mark_read(request: Request, alert_id: int):
    u = current_user(request)
    if not require_role(u, ["coadmin", "admin"]):
        return RedirectResponse("/login", status_code=303)

    mark_alert_read(int(alert_id))

    if u["role"] == "admin":
        return RedirectResponse("/admin", status_code=303)
    return RedirectResponse(f"/coadmin/{u['team']}", status_code=303)