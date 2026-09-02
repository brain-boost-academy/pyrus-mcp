"""Таблица инструментов сверяется с реальным upstream в обе стороны."""

import inspect

import pyrus.models.requests as pyrus_requests
import pytest
from pyrus.client import PyrusAPI

from pyrus_mcp.catalog import ALLOWED, CATALOG, NOT_TOOLS, Level, Mode, Shape, visible


def test_every_spec_method_exists_upstream():
    for spec in CATALOG:
        assert callable(getattr(PyrusAPI, spec.method, None)), spec.method


def test_every_spec_request_class_exists_upstream():
    for spec in CATALOG:
        if spec.request:
            assert inspect.isclass(getattr(pyrus_requests, spec.request, None)), spec.request


def test_every_req_shape_names_a_request_class():
    """Проверка выше сама пропускает строку с request=None, а регистрация её
    не ловит: _spec_keys при пустом spec.request падает в сигнатуру метода,
    и инструмент заводится штатно, чтобы взорваться на первом вызове. Здесь
    проверяется весь каталог сразу, в отличие от assert в _build_request,
    которого нет под python -O."""
    missing = [s.method for s in CATALOG if s.shape in (Shape.REQ, Shape.ID_REQ) and not s.request]
    assert missing == []


def test_no_upstream_method_is_missing_from_the_table():
    """Обратная сверка: новый метод в pyrus-api не должен пройти незамеченным."""
    upstream = {
        name
        for name, _ in inspect.getmembers(PyrusAPI, inspect.isfunction)
        if not name.startswith("_") and name not in NOT_TOOLS
    }
    assert upstream == {s.method for s in CATALOG}


def test_every_mode_is_covered_and_full_sees_every_level():
    """Новый Level, забытый в ALLOWED, не выдал бы ошибку — инструменты этого
    уровня просто исчезли бы из всех режимов, включая full."""
    assert set(ALLOWED) == set(Mode)
    assert ALLOWED[Mode.FULL] == set(Level)


def test_levels_are_distributed_as_designed():
    counts = {level: sum(s.level is level for s in CATALOG) for level in Level}
    assert counts == {Level.READ: 22, Level.WRITE: 17, Level.DESTRUCTIVE: 2}


@pytest.mark.parametrize(
    ("mode", "file_root", "expected"),
    [
        (Mode.READ_ONLY, False, 22),
        (Mode.READ_ONLY, True, 22),
        (Mode.NON_DESTRUCTIVE, False, 38),
        (Mode.NON_DESTRUCTIVE, True, 39),
        (Mode.FULL, False, 40),
        (Mode.FULL, True, 41),
    ],
)
def test_visible_tool_counts(mode, file_root, expected):
    assert len(visible(mode, file_root=file_root)) == expected


def test_read_only_hides_every_writing_tool():
    names = {s.method for s in visible(Mode.READ_ONLY, file_root=True)}
    assert "create_task" not in names
    assert "delete_role" not in names
    assert "upload_file" not in names
    assert "download_file" in names


def test_non_destructive_hides_deletes_but_keeps_sync_catalog():
    names = {s.method for s in visible(Mode.NON_DESTRUCTIVE, file_root=True)}
    assert "delete_role" not in names
    assert "delete_knowledge_base_entity" not in names
    assert "sync_catalog" in names


def _guard(method):
    guard = next(s.guard for s in CATALOG if s.method == method)
    assert guard is not None, method
    return guard


@pytest.mark.parametrize(
    ("method", "payload"),
    [
        ("update_catalog_items", {"delete": ["ключ-строки"]}),
        ("sync_catalog", {"apply": True}),
        ("sync_catalog", {"apply": None}),
        ("sync_catalog", {}),
        ("sync_catalog", {"apply": "false"}),
        ("update_role", {"removed_members": [1]}),
        ("update_role", {"banned": True}),
        ("update_role", {"banned": "true"}),
        ("update_role", {"banned": 1}),
        ("update_member", {"banned": True}),
        ("update_member", {"banned": "true"}),
        ("update_member", {"banned": 1}),
    ],
)
def test_guard_rejects_data_loss(method, payload):
    assert _guard(method)(payload) is not None


@pytest.mark.parametrize(
    ("method", "payload"),
    [
        ("update_catalog_items", {"upsert": [["значение"]]}),
        ("update_catalog_items", {"delete": []}),
        ("sync_catalog", {"apply": False, "items": []}),
        ("update_role", {"name": "новое имя"}),
        ("update_role", {"added_members": [1], "banned": False}),
        ("update_member", {"first_name": "Иван"}),
        ("update_member", {"banned": False}),
    ],
)
def test_guard_passes_safe_payload(method, payload):
    assert _guard(method)(payload) is None


def test_only_four_methods_are_guarded():
    guarded = {s.method for s in CATALOG if s.guard}
    assert guarded == {"update_catalog_items", "sync_catalog", "update_role", "update_member"}


def test_one_offs_are_exactly_the_three_hand_written_tools():
    assert {s.method for s in CATALOG if s.shape is Shape.ONE_OFF} == {
        "download_file",
        "upload_file",
        "set_avatar",
    }
