# DET MONITORING APPLICATION - End User -> Server -> Admin

## What it does
- User page: upload meter image
- Server: runs OCR on uploaded meter images
- Admin page: shows extracted text + uploaded image + raw OCR JSON

## Prerequisites
- Python 3.10+ recommended (Python 3.11 preferred)
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

## If Python crashes on startup
- This usually comes from native OCR libs (`opencv`/`torch`) in a broken virtualenv.
- Use a clean Python 3.11 environment and reinstall dependencies.
- OCR warmup is disabled by default; enable only when stable:
  - `export ENABLE_OCR_WARMUP=1`

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

## Messaging Facility (Popup Chat v2)

### Architecture (text diagram)
```
Browser (User/Coadmin/Admin)
  -> Messaging Popup UI (chat list + thread + composer)
  -> REST APIs (/api/chat/*)
FastAPI (session-auth + role checks)
  -> Chat service logic (membership, blocks, receipts, reactions, reports)
SQLite (chat_conversations, chat_members, chat_messages, chat_receipts, ...)
```

### Security model
- Session-based auth reused from existing app.
- Conversation membership enforced server-side on every chat API.
- Role-scoped user picker (user/coadmin/admin can only start chats with allowed users).
- Block list enforcement for direct chats.
- Message/report APIs require authenticated user and membership.

### Chat DB schema (added)
- `chat_conversations` (`direct` / `group`, metadata, timestamps)
- `chat_members` (member role, pinned, mute, archive, last_read pointer)
- `chat_messages` (text/system, reply id, edit/delete flags, idempotency `client_msg_id`)
- `chat_receipts` (delivered/read per user per message)
- `chat_reactions` (emoji reactions)
- `chat_blocks` (block relationships)
- `chat_reports` (moderation reports)

### Chat APIs
- `GET /api/chat/bootstrap` -> current user + allowed chat user list
- `GET /api/chat/conversations?search=&limit=` -> chat list + unread + preview
- `POST /api/chat/conversations` -> create direct/group conversation
- `GET /api/chat/conversations/{conversation_id}/messages?before_id=&limit=` -> message history
- `POST /api/chat/messages` -> send message
- `POST /api/chat/messages/{message_id}/read` -> mark read
- `POST /api/chat/messages/{message_id}/edit` -> edit own message
- `POST /api/chat/messages/{message_id}/delete` -> soft delete (sender/admin)
- `POST /api/chat/messages/{message_id}/react` -> toggle emoji reaction
- `POST /api/chat/conversations/{conversation_id}/typing` -> typing start/stop
- `POST /api/chat/conversations/{conversation_id}/settings` -> pin/mute
- `POST /api/chat/block` -> block user
- `POST /api/chat/report` -> report message/chat

### WebSocket/Event note
- Current implementation uses short-interval polling for reliable updates in this stack.
- Event equivalents implemented by polling:
  - new message
  - message edited/deleted
  - read state update
  - typing indicator
  - conversation unread/preview updates

### UI behavior
- New `Messaging` button in nav opens popup on user/coadmin/admin pages.
- Popup includes:
  - chat list with unread count + preview
  - direct/group creation
  - thread view with message actions (react/edit/delete/report)
  - typing indicator + polling refresh
  - composer with emoji shortcuts

### Limits / next upgrades
- Media/file upload in chat, push notifications, and full WebSocket fan-out can be added as next phase.
- Full E2EE is not enabled in this phase.

## Production Logging

### Logging stack detected
- Backend: Python + FastAPI
- Database: SQLite
- Existing logging style before upgrade: mostly `print(...)`

### Logging plan (module -> file)
- `auth` -> `logs/auth.log` -> login/logout/bootstrap user events
- `api` -> `logs/api.log` -> request start/end + endpoint lifecycle
- `db` -> `logs/db.log` -> DB connection, query timings, slow query warnings, query failures
- `admin` -> `logs/admin.log` -> alert creation/admin ops
- `chat` -> `logs/chat.log` -> chat create/send/edit/delete/react/read
- `security` -> `logs/security.log` -> block/security events
- `audit` -> `logs/audit.log` -> exports, clears, reports
- `jobs` -> `logs/jobs.log` -> OCR warmup/background jobs
- `system` -> `logs/system.log` -> startup/readiness/system config events
- `errors` -> `logs/errors.log` -> all ERROR-class events across modules

### Added/modified files
- Added: [server/logging_utils.py](/Users/amitkumar/Documents/meter-ocr-app/server/logging_utils.py)
- Modified: [server/app.py](/Users/amitkumar/Documents/meter-ocr-app/server/app.py)
- Modified: [server/db.py](/Users/amitkumar/Documents/meter-ocr-app/server/db.py)

### Environment variables
- `LOG_LEVEL=debug|info|warn|error`
- `LOG_DIR=./logs`
- `LOG_TO_CONSOLE=true|false`
- `LOG_FORMAT=json|pretty`
- `LOG_RETENTION_DAYS=14`
- `LOG_MAX_SIZE_MB=50`
- `TZ=Asia/Kolkata`
- `SERVICE_NAME=det-monitoring-application`
- `DB_SLOW_QUERY_MS=200`
- `LOG_INCLUDE_STACKTRACE=true|false`
- `ENV=dev|prod` (affects defaults)

### Request tracing
- Per request middleware creates/propagates `request_id`.
- Response includes `X-Request-Id`.
- Request context is injected in logs automatically:
  - `request_id`, `method`, `route`, `ip`, `user_id`, `duration_ms`.

### Example usage points implemented
- Auth login: `AUTH_LOGIN_ATTEMPT`, `AUTH_LOGIN_SUCCESS`, `AUTH_LOGIN_FAIL`
- User bootstrap create: `USER_REGISTER_SUCCESS`
- Normal API lifecycle: `API_REQUEST_START`, `API_REQUEST_END`
- DB connection/query: `DB_CONNECTION_OK`, `DB_SLOW_QUERY`, `DB_QUERY_FAIL`
- Global exception handler: `UNHANDLED_EXCEPTION`
- Admin audit: `ADMIN_EXPORT_CSV`, `ADMIN_CLEAR_ALERTS`
- Security: `SECURITY_BLOCK_USER`

### Log format
All entries include:
- `timestamp` (ISO8601 with milliseconds + timezone)
- `level`
- `service`
- `module`
- `event`
- `message`
- request context fields when available
- `error` object on exceptions (`error_type`, `error_message`, optional `stacktrace`)

### Sample JSON log lines
```json
{"timestamp":"2026-03-05T14:42:10.123+05:30","level":"INFO","service":"det-monitoring-application","module":"auth","event":"AUTH_LOGIN_SUCCESS","message":"Login successful","request_id":"f4d0...","user_id":12,"route":"/login","method":"POST","ip":"127.0.0.1"}
{"timestamp":"2026-03-05T14:42:10.355+05:30","level":"INFO","service":"det-monitoring-application","module":"api","event":"API_REQUEST_END","message":"Request completed","request_id":"f4d0...","route":"/api/chat/messages","method":"POST","duration_ms":86.41,"status_code":200}
{"timestamp":"2026-03-05T14:42:10.501+05:30","level":"WARN","service":"det-monitoring-application","module":"db","event":"DB_SLOW_QUERY","message":"Slow database query detected","duration_ms":245.17,"query":"SELECT ..."}
{"timestamp":"2026-03-05T14:42:10.888+05:30","level":"ERROR","service":"det-monitoring-application","module":"errors","event":"UNHANDLED_EXCEPTION","message":"Unhandled server exception","error":{"error_type":"ValueError","error_message":"...","stacktrace":"..."}}
```

### Sensitive data safety checklist
- Passwords/tokens/API keys/OTP/card-like keys are masked by sanitizer.
- Emails and long phone-like identifiers are masked.
- DB logs do not store query parameter values.
- Auth logs avoid raw secrets.

### Rotation & retention
- Size-based rotation per file (`LOG_MAX_SIZE_MB`), rotated files gzipped.
- Old logs pruned by retention window (`LOG_RETENTION_DAYS`).
- Log directory auto-created.

### Dev vs prod behavior
- Dev defaults: `pretty` console logs + `DEBUG` + stacktrace enabled.
- Prod defaults: `json` logs + `INFO` + stacktrace limited unless enabled.

### Quick log viewing commands
```bash
tail -f logs/api.log
tail -f logs/errors.log
tail -f logs/db.log | rg DB_SLOW_QUERY
rg AUTH_LOGIN_FAIL logs/auth.log
```

## Notes
- Uploads stored in: server/uploads/
- SQLite DB stored in: server/data/meter_ocr.db
