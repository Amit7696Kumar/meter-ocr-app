import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, Form, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.concurrency import run_in_threadpool

import bcrypt as pybcrypt
from passlib.hash import pbkdf2_sha256

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
    clear_alerts_admin,
    count_unread_admin,
    count_unread_coadmin,
    get_latest_reading_id_all,
    get_latest_reading_id_team,
    create_message,
    fetch_messages_for_user,
    mark_message_read,
    fetch_users_all,
    fetch_users_by_team,
)

#  use warmup + run_ocr from ocr_engine
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
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", "CHANGE_ME_TO_A_RANDOM_LONG_SECRET"),
    max_age=60 * 60 * 24 * 365 * 10,  # 10 years; logout clears it explicitly
    same_site="lax",
    https_only=False,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=TEMPLATE_DIR)
templates.env.globals["static_version"] = str(int(time.time()))


# -----------------------
# Ideals + helper utils
# -----------------------
def load_ideals():
    if not os.path.exists(IDEAL_PATH):
        print(f"[IDEALS]  Missing {IDEAL_PATH}. Using no rules.", flush=True)
        return {}

    try:
        with open(IDEAL_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                print(f"[IDEALS]  {IDEAL_PATH} is empty. Using no rules.", flush=True)
                return {}
            data = json.loads(raw)
            if not isinstance(data, dict):
                print(f"[IDEALS]  {IDEAL_PATH} is not a JSON object. Using no rules.", flush=True)
                return {}
            return data
    except Exception as e:
        print(f"[IDEALS]  Failed to read/parse {IDEAL_PATH}: {e}", flush=True)
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


def hash_password(password: str) -> str:
    # Use PBKDF2 to avoid passlib<->bcrypt backend compatibility issues.
    return pbkdf2_sha256.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    if stored_hash.startswith("$2"):
        try:
            return pybcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception:
            return False
    try:
        return pbkdf2_sha256.verify(password, stored_hash)
    except Exception:
        return False


def _augment_readings(readings):
    for r in readings:
        raw = r.get("ocr_json") if isinstance(r, dict) else None
        if not raw:
            r["debug_yolo"] = ""
            r["debug_crop"] = ""
            r["debug_prep"] = ""
            r["ocr_obj"] = None
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            r["debug_yolo"] = ""
            r["debug_crop"] = ""
            r["debug_prep"] = ""
            r["ocr_obj"] = None
            continue
        dbg = obj.get("debug_urls") or {}
        r["debug_yolo"] = dbg.get("yolo", "")
        r["debug_crop"] = dbg.get("crop", "")
        r["debug_prep"] = dbg.get("prep", "")
        r["ocr_obj"] = obj
    return readings


def _filter_today_readings(readings):
    today = datetime.now().strftime("%Y-%m-%d")
    out = []
    for r in readings:
        created = (r.get("created_at") if isinstance(r, dict) else None) or ""
        if str(created).startswith(today):
            out.append(r)
    return out


# -----------------------
# Startup
# -----------------------
import threading

@app.on_event("startup")
def startup():
    init_db()

    #  Start server immediately, warm up in background
    def _warm():
        try:
            print("[WARMUP] Starting in background...", flush=True)
            warmup_models()
            print("[WARMUP] Done.", flush=True)
        except Exception as e:
            print(f"[WARMUP]  Failed: {e}", flush=True)

    threading.Thread(target=_warm, daemon=True).start()

    # create users...
    if not get_user_by_username("admin"):
        create_user("admin", hash_password("admin123"), "admin", None)

    for t in range(1, 6):
        uname = f"coadmin{t}"
        if not get_user_by_username(uname):
            create_user(uname, hash_password("coadmin123"), "coadmin", t)

    for t in range(1, 6):
        for i in range(1, 9):
            uname = f"user{t}{i}"
            if not get_user_by_username(uname):
                create_user(uname, hash_password("user123"), "user", t)

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
    if not u or not verify_password(password, u["password_hash"]):
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

    my_readings = _augment_readings(_filter_today_readings(fetch_readings_by_user(int(u["id"]))))
    uploaded = request.query_params.get("uploaded") == "1"
    messages = fetch_messages_for_user(role=u["role"], user_id=int(u["id"]), team=int(u["team"]))
    return templates.TemplateResponse(
        "user.html",
        {"request": request, "user": u, "readings": my_readings, "uploaded": uploaded, "messages": messages},
    )


@app.post("/upload")
async def upload_meter_image(
    request: Request,
    label: str = Form(...),
    meter_type: str = Form(...),
    manual_value: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    u = current_user(request)
    if not require_role(u, ["user"]):
        return RedirectResponse("/login", status_code=303)

    if meter_type not in ["earthing", "temp", "voltage"]:
        return templates.TemplateResponse(
            "user.html",
            {
                "request": request,
                "user": u,
                "readings": _augment_readings(_filter_today_readings(fetch_readings_by_user(int(u["id"])))),
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
                "readings": _augment_readings(_filter_today_readings(fetch_readings_by_user(int(u["id"])))),
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
                "readings": _augment_readings(_filter_today_readings(fetch_readings_by_user(int(u["id"])))),
                "error": "Only image files are allowed.",
            },
            status_code=400,
        )

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    # Save upload
    with open(filepath, "wb") as f:
        f.write(await image.read())

    #  OCR in threadpool so request doesn't feel "stuck"
    debug_id = uuid.uuid4().hex
    try:
        ocr_result = await run_in_threadpool(run_ocr, filepath, debug_id, meter_type)
    except Exception as e:
        print(f"[OCR]  Failed: {e}", flush=True)
        return templates.TemplateResponse(
            "user.html",
            {
                "request": request,
                "user": u,
                "readings": _augment_readings(_filter_today_readings(fetch_readings_by_user(int(u["id"])))),
                "error": f"OCR failed: {e}",
            },
            status_code=500,
        )

    numeric_value: Optional[str] = None
    if ocr_result.get("numeric"):
        numeric_value = ocr_result["numeric"].get("value")

    manual_value = (manual_value or "").strip() or None
    value_for_alert = manual_value or numeric_value

    # Store reading
    rid = insert_reading(
        user_id=int(u["id"]),
        team=int(u["team"]),
        meter_type=meter_type,
        label=label.strip(),
        value=manual_value or numeric_value,
        filename=filename,
        ocr_json=json.dumps(ocr_result, default=str),
        manual_value=manual_value,
    )

    # Alerts logic
    print(f"[ALERT] meter_type={meter_type} numeric_value={value_for_alert}", flush=True)
    is_alert, severity, msg = compare_against_ideal(meter_type, value_for_alert)
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

    return RedirectResponse("/?uploaded=1", status_code=303)


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

    readings = _augment_readings(_filter_today_readings(fetch_readings_by_team(int(team_id))))
    alerts = fetch_alerts_for_coadmin(int(team_id), unread_only=False)
    unread = count_unread_coadmin(int(team_id))
    latest_id = get_latest_reading_id_team(int(team_id))

    messages = fetch_messages_for_user(role=u["role"], user_id=int(u["id"]), team=int(u["team"]))
    users_team = fetch_users_by_team(int(team_id))
    return templates.TemplateResponse(
        "coadmin.html",
        {
            "request": request,
            "user": u,
            "team_id": team_id,
            "readings": readings,
            "alerts": alerts,
            "unread_count": unread,
            "latest_reading_id": latest_id,
            "messages": messages,
            "users_team": users_team,
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

    readings = _augment_readings(_filter_today_readings(fetch_readings_all()))
    alerts = fetch_alerts_for_admin(unread_only=False)
    unread = count_unread_admin()
    latest_id = get_latest_reading_id_all()

    messages = fetch_messages_for_user(role=u["role"], user_id=int(u["id"]), team=None)
    all_users = fetch_users_all()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "user": u,
            "readings": readings,
            "alerts": alerts,
            "unread_count": unread,
            "teams": [1, 2, 3, 4, 5],
            "latest_reading_id": latest_id,
            "messages": messages,
            "all_users": all_users,
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


@app.get("/api/readings/admin/latest")
def api_latest_reading_admin(request: Request):
    u = current_user(request)
    if not require_role(u, ["admin"]):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {"latest_id": get_latest_reading_id_all()}


@app.get("/api/readings/coadmin/latest")
def api_latest_reading_coadmin(request: Request, team: int):
    u = current_user(request)
    if not require_role(u, ["coadmin", "admin"]):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if u["role"] == "coadmin" and int(u["team"]) != int(team):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return {"latest_id": get_latest_reading_id_team(int(team))}


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


@app.post("/alerts/clear/admin")
async def clear_admin_alerts(request: Request):
    u = current_user(request)
    if not require_role(u, ["admin"]):
        return RedirectResponse("/login", status_code=303)
    clear_alerts_admin()
    return RedirectResponse("/admin", status_code=303)


@app.post("/messages/send")
async def send_message(
    request: Request,
    target_role: str = Form(...),
    target_team: Optional[str] = Form(None),
    target_username: Optional[str] = Form(None),
    body: str = Form(...),
):
    u = current_user(request)
    if not require_role(u, ["user", "coadmin", "admin"]):
        return RedirectResponse("/login", status_code=303)

    target_role = target_role.strip().lower()
    if target_role not in ["user", "coadmin", "admin"]:
        return RedirectResponse("/", status_code=303)

    target_user_id = None
    if target_username:
        user_obj = get_user_by_username(target_username.strip())
        if user_obj:
            target_user_id = int(user_obj["id"])
            target_team = user_obj.get("team")

    # Role-based restrictions
    if u["role"] == "user" and target_role not in ["coadmin", "admin"]:
        return RedirectResponse("/", status_code=303)
    if u["role"] == "coadmin":
        if target_role == "user":
            # restrict to same team unless explicit user
            if target_team is None:
                target_team = int(u["team"])
            if int(target_team) != int(u["team"]):
                return RedirectResponse(f"/coadmin/{u['team']}", status_code=303)
        elif target_role != "admin":
            return RedirectResponse(f"/coadmin/{u['team']}", status_code=303)

    team_val = None
    if target_team is not None and str(target_team).strip() != "":
        try:
            team_val = int(target_team)
        except Exception:
            team_val = None

    create_message(
        sender_user_id=int(u["id"]),
        sender_role=u["role"],
        sender_team=int(u["team"]) if u.get("team") is not None else None,
        target_role=target_role,
        target_team=team_val,
        target_user_id=target_user_id,
        body=body.strip(),
    )

    if u["role"] == "admin":
        return RedirectResponse("/admin", status_code=303)
    if u["role"] == "coadmin":
        return RedirectResponse(f"/coadmin/{u['team']}", status_code=303)
    return RedirectResponse("/", status_code=303)


@app.post("/messages/{message_id}/read")
async def mark_message_as_read(request: Request, message_id: int):
    u = current_user(request)
    if not require_role(u, ["user", "coadmin", "admin"]):
        return RedirectResponse("/login", status_code=303)
    mark_message_read(int(message_id))
    if u["role"] == "admin":
        return RedirectResponse("/admin", status_code=303)
    if u["role"] == "coadmin":
        return RedirectResponse(f"/coadmin/{u['team']}", status_code=303)
    return RedirectResponse("/", status_code=303)
