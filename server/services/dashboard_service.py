from typing import Any, Dict, List, Optional


def build_admin_dashboard_context(
    *,
    user: Dict[str, Any],
    readings: List[Dict[str, Any]],
    task_instances: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    unread_count: int,
    latest_reading_id: int,
    messages: List[Dict[str, Any]],
    all_users: List[Dict[str, Any]],
    teams: Optional[List[int]] = None,
) -> Dict[str, Any]:
    return {
        "user": user,
        "readings": readings,
        "task_instances": task_instances,
        "alerts": alerts,
        "unread_count": unread_count,
        "teams": teams if teams is not None else [1, 2, 3, 4, 5, 6],
        "latest_reading_id": latest_reading_id,
        "messages": messages,
        "all_users": all_users,
    }


def build_coadmin_dashboard_context(
    *,
    user: Dict[str, Any],
    team_id: int,
    readings: List[Dict[str, Any]],
    task_instances: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    unread_count: int,
    latest_reading_id: int,
    messages: List[Dict[str, Any]],
    users_team: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "user": user,
        "team_id": team_id,
        "readings": readings,
        "task_instances": task_instances,
        "alerts": alerts,
        "unread_count": unread_count,
        "latest_reading_id": latest_reading_id,
        "messages": messages,
        "users_team": users_team,
        "total_task_count": len(readings) + len(task_instances),
    }
