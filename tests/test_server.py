"""Сервер целиком офлайн: ни сети, ни HTTP-моков."""

import asyncio
import json
from dataclasses import replace

import pytest
from conftest import FakeClient, FakeResponse
from fastmcp import Client
from fastmcp.exceptions import ToolError

from pyrus_mcp import server
from pyrus_mcp.catalog import CATALOG, Mode, Shape

# --- режим -----------------------------------------------------------------


def test_mode_defaults_to_non_destructive(monkeypatch):
    monkeypatch.delenv("PYRUS_MCP_MODE", raising=False)
    assert server.resolve_mode([]) is Mode.NON_DESTRUCTIVE


def test_mode_reads_env(monkeypatch):
    monkeypatch.setenv("PYRUS_MCP_MODE", "read-only")
    assert server.resolve_mode([]) is Mode.READ_ONLY


def test_flag_beats_env(monkeypatch):
    monkeypatch.setenv("PYRUS_MCP_MODE", "read-only")
    assert server.resolve_mode(["--mode", "full"]) is Mode.FULL


def test_invalid_mode_exits_and_lists_valid_ones(monkeypatch, capsys):
    monkeypatch.delenv("PYRUS_MCP_MODE", raising=False)
    with pytest.raises(SystemExit) as exc:
        server.resolve_mode(["--mode", "bogus"])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "read-only" in err and "non-destructive" in err and "full" in err


# --- fail-fast на старте ---------------------------------------------------


def _env(monkeypatch, **extra):
    monkeypatch.delenv("PYRUS_MCP_MODE", raising=False)
    monkeypatch.delenv("PYRUS_MCP_FILE_ROOT", raising=False)
    monkeypatch.setenv("PYRUS_LOGIN", "u@example.com")
    monkeypatch.setenv("PYRUS_SECURITY_KEY", "k")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


def test_missing_credentials_exit_before_serving(monkeypatch):
    monkeypatch.delenv("PYRUS_LOGIN", raising=False)
    monkeypatch.delenv("PYRUS_SECURITY_KEY", raising=False)
    monkeypatch.delenv("PYRUS_MCP_MODE", raising=False)
    with pytest.raises(SystemExit) as exc:
        server.main([])
    assert exc.value.code != 0


def test_auth_that_returns_an_error_object_still_fails_fast(monkeypatch):
    """auth() не бросает исключений — она ВОЗВРАЩАЕТ объект с error_code.
    try/except вокруг неё не поймал бы ничего, и сервер стартовал бы
    с мёртвыми ключами."""
    _env(monkeypatch)
    started = []
    fake = FakeClient(result=FakeResponse(error_code="authorization_error"))
    monkeypatch.setattr(server, "_make_client", lambda **kw: fake)
    monkeypatch.setattr(server.FastMCP, "run", lambda self, *a, **k: started.append(1))

    with pytest.raises(SystemExit) as exc:
        server.main([])
    assert exc.value.code != 0
    assert started == []


def test_successful_auth_reaches_run(monkeypatch, capsys):
    _env(monkeypatch)
    started = []
    monkeypatch.setattr(server, "_make_client", lambda **kw: FakeClient())
    monkeypatch.setattr(server.FastMCP, "run", lambda self, *a, **k: started.append(1))
    server.main([])
    assert started == [1]
    captured = capsys.readouterr()
    assert captured.out == "", "stdout занят MCP-протоколом"
    assert "режим non-destructive" in captured.err


# --- адаптер результата ----------------------------------------------------


def test_error_code_becomes_a_tool_error():
    with pytest.raises(ToolError, match="access_denied"):
        server._adapt(FakeResponse(error_code="access_denied"), 0, 1000)


def test_raw_api_payload_is_returned_verbatim():
    assert server._adapt(FakeResponse({"tasks": [{"id": 1}]}), 0, 1000) == {"tasks": [{"id": 1}]}


def test_object_without_raw_payload_falls_back_to_jsonpickle():
    class Nested:
        def __init__(self):
            self.csv = "a;b\n1;2"
            self.error_code = None
            self.original_response = {}

    assert server._adapt(Nested(), 0, 1000) == {"csv": "a;b\n1;2", "error_code": None,
                                                "original_response": {}}


def test_oversized_result_returns_a_window_not_silent_truncation():
    payload = {"tasks": [{"id": i, "text": "x" * 100} for i in range(200)]}
    out = server._adapt(FakeResponse(payload), 0, 4096)
    page = out["_pagination"]
    assert page["total"] == 200
    assert page["returned"] < 200
    assert page["next_offset"] == page["returned"]
    assert out["tasks"] == payload["tasks"][: page["returned"]]


def test_window_walks_to_the_end():
    payload = {"tasks": [{"id": i, "text": "x" * 100} for i in range(200)]}
    seen, offset = 0, 0
    while offset is not None:
        out = server._adapt(FakeResponse(payload), offset, 4096)
        seen += out["_pagination"]["returned"]
        offset = out["_pagination"]["next_offset"]
    assert seen == 200


def test_oversized_result_without_a_sliceable_list_says_so():
    with pytest.raises(ToolError, match="списка"):
        server._adapt(FakeResponse({"blob": "x" * 5000}), 0, 1024)


def test_negative_offset_is_rejected_not_silently_windowed():
    """items[-5:] отдал бы хвост списка под видом окна с начала, а next_offset
    отправил бы модель на второй проход по тем же данным."""
    payload = {"tasks": [{"id": i} for i in range(10)]}
    with pytest.raises(ToolError, match="отрицательным"):
        server._adapt(FakeResponse(payload), -5, 1_000_000)


def test_offset_on_a_listless_response_returns_the_result_not_a_retry_order():
    """get_task возвращает {"task": {...}} без списка. Прежний текст ошибки
    приказывал «повторите вызов с offset=0», и на пишущем инструменте это был
    приказ повторить незаидемпотентную запись. Результат отдаётся с пометкой."""
    out = server._adapt(FakeResponse({"task": {"id": 1}}), 5, 1_000_000)
    assert out["task"] == {"id": 1}
    assert out["_pagination"]["field"] is None
    assert out["_pagination"]["offset"] == 5
    assert "повторять его не нужно" in out["_pagination"]["note"]


def test_listless_response_over_the_limit_still_blames_the_size():
    """Пометкой тут не отделаться: отдать нечего, ответ не влезает."""
    with pytest.raises(ToolError, match="не содержит"):
        server._adapt(FakeResponse({"blob": "x" * 5000}), 5, 1024)


def test_offset_past_the_end_keeps_the_pagination_contract():
    payload = {"tasks": [{"id": i} for i in range(10)]}
    page = server._adapt(FakeResponse(payload), 99, 1_000_000)["_pagination"]
    assert page == {"field": "tasks", "offset": 99, "returned": 0, "total": 10,
                    "next_offset": None}


# --- диспетчер -------------------------------------------------------------


def _spec(method):
    return next(s for s in CATALOG if s.method == method)


def test_unknown_request_key_gives_valid_keys_not_a_bare_typeerror():
    with pytest.raises(ToolError) as exc:
        server._build_request(_spec("create_task"), {"txet": "опечатка"})
    assert "text" in str(exc.value) and "subject" in str(exc.value)


def test_missing_required_key_is_named():
    with pytest.raises(ToolError) as exc:
        server._build_request(_spec("get_calendar_tasks"), {})
    assert "start_date_utc" in str(exc.value)


def test_kwargs_request_classes_still_expose_key_list():
    """CreateMemberRequest — **kwargs над Person, inspect.signature по нему
    пуст. Ключи всё равно должны попасть в описание инструмента."""
    keys, _ = server._spec_keys(_spec("update_member"))
    assert "first_name" in keys and "email" in keys


def test_unknown_option_key_gives_valid_keys():
    """Настоящий PyrusAPI: TypeError летит из сигнатуры до любого HTTP-вызова."""
    from pyrus.client import PyrusAPI

    with pytest.raises(ToolError) as exc:
        server._call_plain(PyrusAPI(), _spec("get_catalog"), [1], {"filterz": 1})
    assert "filters" in str(exc.value)


@pytest.mark.parametrize(
    ("var", "value", "hint"),
    [
        ("PYRUS_MCP_MAX_RESULT_BYTES", "много", "целым числом"),
        ("PYRUS_MCP_MAX_RESULT_BYTES", "0", "положительным"),
        ("PYRUS_PERSON_ID", "abc", "целым числом"),
    ],
)
def test_broken_numeric_env_exits_cleanly_instead_of_a_traceback(monkeypatch, capsys, var, value, hint):
    _env(monkeypatch, **{var: value})
    with pytest.raises(SystemExit) as exc:
        server.main([])
    assert exc.value.code != 0
    assert hint in capsys.readouterr().err


@pytest.mark.parametrize("mode", [Mode.READ_ONLY, Mode.NON_DESTRUCTIVE, Mode.FULL])
def test_guards_only_bite_outside_full(mode):
    spec = _spec("sync_catalog")
    if mode is Mode.FULL:
        server._check_guard(spec, {"apply": True}, mode)  # не бросает
    else:
        with pytest.raises(ToolError, match="mode full"):
            server._check_guard(spec, {"apply": True}, mode)


# --- регистрация и tools/list ----------------------------------------------


def _build(client, mode, file_root=None, max_bytes=65536):
    return server.build(client, mode, file_root, max_bytes)


def _tool_names(mcp):
    async def go():
        async with Client(mcp) as c:
            return {t.name for t in await c.list_tools()}

    return asyncio.run(go())


def _call(mcp, name, args):
    async def go():
        async with Client(mcp) as c:
            return await c.call_tool(name, args)

    return asyncio.run(go())


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(Mode.READ_ONLY, 22), (Mode.NON_DESTRUCTIVE, 38), (Mode.FULL, 40)],
)
def test_tools_list_matches_mode_without_file_root(mode, expected):
    assert len(_tool_names(_build(FakeClient(), mode))) == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(Mode.READ_ONLY, 22), (Mode.NON_DESTRUCTIVE, 39), (Mode.FULL, 41)],
)
def test_tools_list_matches_mode_with_file_root(mode, expected, tmp_path):
    assert len(_tool_names(_build(FakeClient(), mode, tmp_path.resolve()))) == expected


def test_read_only_has_no_writing_tool_in_tools_list():
    names = _tool_names(_build(FakeClient(), Mode.READ_ONLY))
    assert not names & {"create_task", "delete_role", "sync_catalog", "upload_file"}
    assert "download_file" in names


def test_non_destructive_omits_deletes_from_tools_list():
    names = _tool_names(_build(FakeClient(), Mode.NON_DESTRUCTIVE))
    assert not names & {"delete_role", "delete_knowledge_base_entity"}
    assert "sync_catalog" in names


def test_instructions_name_the_active_mode():
    mcp = _build(FakeClient(), Mode.READ_ONLY)
    assert "read-only" in mcp.instructions


def test_every_registered_tool_has_a_description():
    async def go():
        async with Client(_build(FakeClient(), Mode.FULL)) as c:
            return await c.list_tools()

    for tool in asyncio.run(go()):
        assert tool.description, tool.name


# --- формы сигнатур вызывают upstream правильно ----------------------------


def test_plain_shape_forwards_options():
    client = FakeClient()
    _call(_build(client, Mode.READ_ONLY), "get_announcements", {"options": {"item_count": 5}})
    assert client.calls == [("get_announcements", (), {"item_count": 5})]


def test_by_id_shape_passes_id_positionally():
    client = FakeClient()
    _call(_build(client, Mode.READ_ONLY), "get_catalog", {"entity_id": 7})
    assert client.calls == [("get_catalog", (7,), {})]


def test_req_shape_builds_the_request_object():
    client = FakeClient()
    _call(_build(client, Mode.NON_DESTRUCTIVE), "create_task", {"request": {"text": "привет"}})
    name, args, _ = client.calls[0]
    assert name == "create_task"
    assert args[0].text == "привет"


def test_id_req_shape_passes_id_then_request():
    client = FakeClient()
    _call(
        _build(client, Mode.NON_DESTRUCTIVE),
        "comment_task",
        {"entity_id": 42, "request": {"text": "ок"}},
    )
    name, args, _ = client.calls[0]
    assert name == "comment_task"
    assert args[0] == 42
    assert args[1].text == "ок"


def test_guard_blocks_the_call_before_it_reaches_pyrus():
    client = FakeClient()
    mcp = _build(client, Mode.NON_DESTRUCTIVE)
    with pytest.raises(ToolError, match="delete"):
        _call(mcp, "update_catalog_items", {"entity_id": 1, "request": {"delete": ["ключ-строки"]}})
    assert client.calls == []


def test_same_call_passes_with_only_upsert():
    client = FakeClient()
    mcp = _build(client, Mode.NON_DESTRUCTIVE)
    _call(mcp, "update_catalog_items", {"entity_id": 1, "request": {"upsert": [["значение"]]}})
    assert client.calls[0][0] == "update_catalog_items"


def test_guard_also_fires_on_option_carrying_shapes():
    """Guard вызывался только в формах REQ/ID_REQ. Сегодня все четыре guard'а
    стоят на ID_REQ, но guard, повешенный на BY_ID, молча не срабатывал бы —
    и таблица врала бы о защите."""
    spec = replace(
        _spec("get_catalog"), guard=lambda p: "'filters'" if p.get("filters") else None
    )
    client = FakeClient()
    mcp = server.FastMCP(name="t")
    server._register(mcp, client, spec, Mode.NON_DESTRUCTIVE, 65536)
    with pytest.raises(ToolError, match="filters"):
        _call(mcp, "get_catalog", {"entity_id": 1, "options": {"filters": {"a": 1}}})
    assert client.calls == []


def test_full_mode_lets_the_guarded_field_through():
    client = FakeClient()
    mcp = _build(client, Mode.FULL)
    _call(mcp, "update_catalog_items", {"entity_id": 1, "request": {"delete": ["ключ-строки"]}})
    assert client.calls[0][0] == "update_catalog_items"


def test_offset_is_offered_everywhere_including_writing_tools():
    """Пишущие инструменты тоже отдают списки: sync_catalog в non-destructive
    живёт только как dry-run «верни весь дифф», и без offset второе окно
    диффа недостижимо. Проверка required ловит и схлопывание схемы целиком."""

    async def go():
        async with Client(_build(FakeClient(), Mode.FULL)) as c:
            return {t.name: t.input_schema for t in await c.list_tools()}

    schemas = asyncio.run(go())
    props = {n: s.get("properties", {}) for n, s in schemas.items()}
    assert "offset" in props["get_registry"]
    assert "offset" in props["sync_catalog"]
    assert set(props["create_task"]) == {"request", "offset"}
    assert schemas["create_task"].get("required") == ["request"]


def test_offset_reaches_pagination_through_the_tool():
    """Единственное непроверенное звено offset-контракта: провода от параметра
    инструмента до _adapt. Остальные проверки зовут _adapt/_paginate напрямую."""
    payload = {"tasks": [{"id": i} for i in range(10)]}
    out = _call(
        _build(FakeClient(result=FakeResponse(payload)), Mode.READ_ONLY),
        "get_forms",
        {"offset": 5},
    )
    assert out.data["_pagination"] == {
        "field": "tasks",
        "offset": 5,
        "returned": 5,
        "total": 10,
        "next_offset": None,
    }


def test_offset_reaches_pagination_through_a_writing_tool():
    """Каждая из семи ветвей match протаскивает offset в _adapt отдельно.
    Проверка на PLAIN покрывает одну; здесь ID_REQ — та самая форма, ради
    диффа которой offset и оставлен видимым у пишущих инструментов."""
    payload = {"items": [{"id": i} for i in range(10)]}
    out = _call(
        _build(FakeClient(result=FakeResponse(payload)), Mode.NON_DESTRUCTIVE),
        "sync_catalog",
        {"entity_id": 1, "request": {"apply": False}, "offset": 5},
    )
    assert out.data["_pagination"]["offset"] == 5
    assert out.data["_pagination"]["returned"] == 5


def test_negative_offset_is_refused_before_the_write_happens():
    """Раньше method() уходил в Pyrus, и только потом _paginate смотрел на
    offset: задача создавалась, её id выбрасывался, модель получала претензию
    к своему аргументу. Проверка порядка, а не текста ошибки."""
    client = FakeClient(result=FakeResponse({"task": {"id": 42}}))
    with pytest.raises(ToolError, match="отрицательным"):
        _call(
            _build(client, Mode.NON_DESTRUCTIVE),
            "create_task",
            {"request": {"text": "x"}, "offset": -1},
        )
    assert client.calls == []


def test_positive_offset_on_a_write_does_not_order_a_second_write():
    """create_task отвечает без списка верхнего уровня. Модель обязана получить
    id созданной задачи, а не приказ повторить вызов."""
    client = FakeClient(result=FakeResponse({"task": {"id": 42}}))
    out = _call(
        _build(client, Mode.NON_DESTRUCTIVE),
        "create_task",
        {"request": {"text": "x"}, "offset": 5},
    )
    assert out.data["task"] == {"id": 42}
    assert out.data["_pagination"]["field"] is None
    assert [c[0] for c in client.calls] == ["create_task"]


# --- download_file: инлайн, без записи на диск -----------------------------


class _Download:
    error_code = None

    def __init__(self, filename, raw):
        self.filename, self.raw_file = filename, raw


def test_download_file_returns_text_inline():
    client = FakeClient(result=_Download("отчёт.txt", "привет".encode()))
    out = _call(_build(client, Mode.READ_ONLY), "download_file", {"file_id": 3})
    assert out.data == {"filename": "отчёт.txt", "encoding": "utf-8", "content": "привет"}


def test_download_file_base64s_binary():
    client = FakeClient(result=_Download("a.bin", b"\xff\xfe\x00"))
    out = _call(_build(client, Mode.READ_ONLY), "download_file", {"file_id": 3})
    assert out.data["encoding"] == "base64"
    assert out.data["content"] == "//4A"


def test_download_file_refuses_to_blow_the_context():
    client = FakeClient(result=_Download("big.bin", b"x" * 10_000))
    with pytest.raises(ToolError, match="превышает лимит"):
        _call(_build(client, Mode.READ_ONLY, max_bytes=1024), "download_file", {"file_id": 3})


# --- песочница upload_file -------------------------------------------------


def test_upload_file_absent_without_file_root():
    assert "upload_file" not in _tool_names(_build(FakeClient(), Mode.FULL))


def test_upload_file_present_with_file_root(tmp_path):
    assert "upload_file" in _tool_names(_build(FakeClient(), Mode.FULL, tmp_path.resolve()))


def test_upload_file_accepts_a_path_inside_the_root(tmp_path):
    (tmp_path / "ok.txt").write_text("x")
    client = FakeClient()
    _call(_build(client, Mode.FULL, tmp_path.resolve()), "upload_file", {"file_path": "ok.txt"})
    assert client.calls[0][0] == "upload_file"
    assert client.calls[0][1][0] == str(tmp_path.resolve() / "ok.txt")


@pytest.mark.parametrize("bad", ["../outside.txt", "/etc/passwd", "sub/../../outside.txt"])
def test_upload_file_rejects_traversal(tmp_path, bad):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("секрет")
    client = FakeClient()
    with pytest.raises(ToolError, match="вне разрешённого корня"):
        _call(_build(client, Mode.FULL, root.resolve()), "upload_file", {"file_path": bad})
    assert client.calls == []


def test_upload_file_rejects_a_symlink_pointing_out(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "outside.txt"
    secret.write_text("секрет")
    (root / "link.txt").symlink_to(secret)
    client = FakeClient()
    with pytest.raises(ToolError, match="вне разрешённого корня"):
        _call(_build(client, Mode.FULL, root.resolve()), "upload_file", {"file_path": "link.txt"})
    assert client.calls == []


def test_set_avatar_takes_a_guid_not_a_path():
    client = FakeClient()
    _call(
        _build(client, Mode.NON_DESTRUCTIVE),
        "set_avatar",
        {"member_id": 5, "file_guid": "abc-123"},
    )
    assert client.calls == [("set_avatar", (5, "abc-123", None), {})]


# --- чистота stdout --------------------------------------------------------


def test_registration_and_calls_write_nothing_to_stdout(capsys):
    client = FakeClient()
    mcp = _build(client, Mode.FULL)
    _call(mcp, "get_forms", {})
    assert capsys.readouterr().out == ""


def test_every_tool_result_is_json_serialisable():
    client = FakeClient(result=FakeResponse({"a": [1, 2]}))
    out = _call(_build(client, Mode.READ_ONLY), "get_forms", {})
    json.dumps(out.data)


def test_one_off_shapes_are_not_registered_by_the_generic_path():
    assert all(s.shape is not Shape.ONE_OFF or s.request is None for s in CATALOG)
