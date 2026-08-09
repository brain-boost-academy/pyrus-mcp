"""Таблица инструментов: уровень доступа, форма сигнатуры, guard'ы payload.

Единственный источник правды о том, какой метод PyrusAPI на каком уровне живёт.
Имена методов и классов запросов не дублируются нигде больше.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class Level(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class Mode(StrEnum):
    READ_ONLY = "read-only"
    NON_DESTRUCTIVE = "non-destructive"
    FULL = "full"


ALLOWED: dict[Mode, frozenset[Level]] = {
    Mode.READ_ONLY: frozenset({Level.READ}),
    Mode.NON_DESTRUCTIVE: frozenset({Level.READ, Level.WRITE}),
    Mode.FULL: frozenset({Level.READ, Level.WRITE, Level.DESTRUCTIVE}),
}


class Shape(StrEnum):
    PLAIN = "plain"  # method(**options)
    BY_ID = "by_id"  # method(entity_id, **options)
    REQ = "req"  # method(ReqCls(**request))
    ID_REQ = "id_req"  # method(entity_id, ReqCls(**request))
    ONE_OFF = "one_off"  # своя обёртка в server.py


@dataclass(frozen=True, slots=True)
class Spec:
    method: str
    level: Level
    shape: Shape
    request: str | None = None  # имя класса из pyrus.models.requests
    guard: Callable[[dict], str | None] | None = None
    id_arg: str = "entity_id"  # человекочитаемое имя первого позиционного аргумента


# --- guard'ы payload -------------------------------------------------------
# Возвращают описание отклонённого поля или None. Все булевы проверки —
# allowlist: проходит только явное False/отсутствие, а не «всё, кроме True».
# Иначе {"banned": "true"} и {"banned": 1} проезжают мимо запрета.


def _guard_update_catalog_items(payload: dict) -> str | None:
    if payload.get("delete"):
        return "'delete' (удаление строк справочника)"
    return None


def _guard_sync_catalog(payload: dict) -> str | None:
    if payload.get("apply") is not False:
        return (
            "'apply' в любом значении, кроме явного false — синхронизация с apply=true "
            "удаляет строки, которых нет во входных данных, а пропущенный ключ "
            "трактуется сервером Pyrus как apply=true"
        )
    return None


def _guard_update_role(payload: dict) -> str | None:
    if payload.get("removed_members"):
        return "'removed_members' (исключение участников из роли)"
    if payload.get("banned") not in (None, False):
        return "'banned' (блокировка роли)"
    return None


def _guard_update_member(payload: dict) -> str | None:
    if payload.get("banned") not in (None, False):
        return "'banned' (блокировка пользователя)"
    return None


# --- таблица ---------------------------------------------------------------

CATALOG: tuple[Spec, ...] = (
    # READ — 22
    Spec("get_forms", Level.READ, Shape.PLAIN),
    Spec("get_form", Level.READ, Shape.BY_ID, id_arg="form_id"),
    Spec("get_registry", Level.READ, Shape.ID_REQ, "FormRegisterRequest", id_arg="form_id"),
    Spec("get_task", Level.READ, Shape.BY_ID, id_arg="task_id"),
    Spec("get_task_list", Level.READ, Shape.ID_REQ, "TaskListRequest", id_arg="list_id"),
    Spec("get_lists", Level.READ, Shape.PLAIN),
    Spec("get_contacts", Level.READ, Shape.PLAIN),
    Spec("get_catalog", Level.READ, Shape.BY_ID, id_arg="catalog_id"),
    Spec("get_announcement", Level.READ, Shape.BY_ID, id_arg="announcement_id"),
    Spec("get_announcements", Level.READ, Shape.PLAIN),
    Spec("get_roles", Level.READ, Shape.PLAIN),
    Spec("get_role", Level.READ, Shape.BY_ID, id_arg="role_id"),
    Spec("get_members", Level.READ, Shape.PLAIN),
    Spec("get_member", Level.READ, Shape.BY_ID, id_arg="member_id"),
    Spec("get_profile", Level.READ, Shape.PLAIN),
    Spec("get_inbox", Level.READ, Shape.PLAIN),
    Spec("get_calendar_tasks", Level.READ, Shape.REQ, "CalendarRequest"),
    Spec("get_form_permissions", Level.READ, Shape.BY_ID, id_arg="form_id"),
    Spec("get_knowledge_base_entity", Level.READ, Shape.BY_ID),
    Spec("get_knowledge_base_structure", Level.READ, Shape.PLAIN),
    Spec("get_knowledge_base_permissions", Level.READ, Shape.BY_ID),
    Spec("download_file", Level.READ, Shape.ONE_OFF),
    # WRITE — 17
    Spec("create_task", Level.WRITE, Shape.REQ, "CreateTaskRequest"),
    Spec("comment_task", Level.WRITE, Shape.ID_REQ, "TaskCommentRequest", id_arg="task_id"),
    Spec("create_announcement", Level.WRITE, Shape.REQ, "CreateAnnouncementRequest"),
    Spec(
        "comment_announcement",
        Level.WRITE,
        Shape.ID_REQ,
        "AnnouncementCommentRequest",
        id_arg="announcement_id",
    ),
    Spec("upload_file", Level.WRITE, Shape.ONE_OFF),
    Spec("create_catalog", Level.WRITE, Shape.REQ, "CreateCatalogRequest"),
    Spec(
        "update_catalog_items",
        Level.WRITE,
        Shape.ID_REQ,
        "UpdateCatalogItemsRequest",
        _guard_update_catalog_items,
        id_arg="catalog_id",
    ),
    Spec(
        "sync_catalog",
        Level.WRITE,
        Shape.ID_REQ,
        "SyncCatalogRequest",
        _guard_sync_catalog,
        id_arg="catalog_id",
    ),
    Spec("create_role", Level.WRITE, Shape.REQ, "CreateRoleRequest"),
    Spec(
        "update_role",
        Level.WRITE,
        Shape.ID_REQ,
        "UpdateRoleRequest",
        _guard_update_role,
        id_arg="role_id",
    ),
    Spec("create_member", Level.WRITE, Shape.REQ, "CreateMemberRequest"),
    Spec(
        "update_member",
        Level.WRITE,
        Shape.ID_REQ,
        "UpdateMemberRequest",
        _guard_update_member,
        id_arg="member_id",
    ),
    Spec("set_avatar", Level.WRITE, Shape.ONE_OFF),
    Spec(
        "change_form_permissions",
        Level.WRITE,
        Shape.ID_REQ,
        "ChangePermissionsRequest",
        id_arg="form_id",
    ),
    Spec(
        "create_knowledge_base_entity",
        Level.WRITE,
        Shape.REQ,
        "CreateKnowledgeBaseEntityRequest",
    ),
    Spec(
        "update_knowledge_base_entity",
        Level.WRITE,
        Shape.ID_REQ,
        "UpdateKnowledgeBaseEntityRequest",
    ),
    Spec(
        "update_knowledge_base_permissions",
        Level.WRITE,
        Shape.ID_REQ,
        "UpdateKnowledgeBasePermissionsRequest",
    ),
    # DESTRUCTIVE — 2
    Spec("delete_role", Level.DESTRUCTIVE, Shape.ID_REQ, "DeleteRoleRequest", id_arg="role_id"),
    Spec("delete_knowledge_base_entity", Level.DESTRUCTIVE, Shape.BY_ID),
)

# Методы PyrusAPI, которые сознательно не становятся инструментами.
# auth — вызывается сервером на старте; serialize_request — внутренняя утилита;
# PYRUS_*_URL — константы, объявленные как методы.
NOT_TOOLS = frozenset({"auth", "serialize_request", "PYRUS_API_URL", "PYRUS_AUTH_URL"})

# upload_file читает локальный файл, поэтому регистрируется только при
# заданном PYRUS_MCP_FILE_ROOT. download_file ничего не пишет на диск —
# он возвращает содержимое инлайн и в песочнице не нуждается.
NEEDS_FILE_ROOT = frozenset({"upload_file"})


def visible(mode: Mode, *, file_root: bool) -> tuple[Spec, ...]:
    """Инструменты, которые будут зарегистрированы. Фильтр применяется на
    этапе регистрации: запрещённого инструмента в tools/list просто нет."""
    return tuple(
        s
        for s in CATALOG
        if s.level in ALLOWED[mode] and (file_root or s.method not in NEEDS_FILE_ROOT)
    )
