"""Сервер целиком офлайн: ни сети, ни HTTP-моков."""

import asyncio
import json
from dataclasses import replace
from typing import cast

import pytest
import requests
from conftest import FakeClient, FakeResponse
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pyrus.client import PyrusAPI

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


def test_listless_write_over_the_limit_is_a_result_not_an_error():
    """comment_task на большой задаче отвечает всей задачей без списка верхнего
    уровня. Ошибка (isError) после состоявшейся записи читалась бы клиентом или
    моделью как «не получилось» и вела к дублю, поэтому отдаётся результат."""
    big = FakeResponse({"task": {"id": 1, "text": "x" * 5000}})
    client = FakeClient(result=big)
    out = _call(
        _build(client, Mode.NON_DESTRUCTIVE, max_bytes=1024),
        "comment_task",
        {"entity_id": 1, "request": {"formatted_text": "<b>x</b>"}},
    ).data
    assert "Вызов выполнен, изменение уже применено" in out["_pagination"]["note"]
    assert out["_pagination"]["next_offset"] is None
    assert "task" not in out
    assert [c[0] for c in client.calls] == ["comment_task"]


def test_listless_write_over_the_limit_keeps_ids_and_mentions():
    """Подсказка обещает mentions в ответе этого же вызова, а get_task по той же
    задаче упрётся в тот же лимит: id и mentions должны выжить в _receipt."""
    comments = [
        {"id": 10, "text": "x" * 3000},
        {"id": 11, "text": "y" * 3000, "mentions": [1318562]},
    ]
    client = FakeClient(result=FakeResponse({"task": {"id": 7, "comments": comments}}))
    out = _call(
        _build(client, Mode.NON_DESTRUCTIVE, max_bytes=1024),
        "comment_task",
        {"entity_id": 7, "request": {"formatted_text": "<b>x</b>"}},
    ).data
    assert out["_receipt"] == {"task_id": 7, "last_comment": {"id": 11, "mentions": [1318562]}}


def test_listless_create_over_the_limit_keeps_the_new_id():
    client = FakeClient(result=FakeResponse({"announcement": {"id": 5, "text": "x" * 5000}}))
    out = _call(
        _build(client, Mode.NON_DESTRUCTIVE, max_bytes=1024),
        "create_announcement",
        {"request": {"text": "x"}},
    ).data
    assert out["_receipt"] == {"announcement_id": 5}


def test_receipt_keeps_a_top_level_id_under_its_own_name():
    """Квитанция без id после «не повторяйте вызов» теряла бы созданную запись."""
    data = {"catalog_id": 3, "blob": "x" * 5000}
    out = server._adapt(FakeResponse(data), 0, 1024, write=True)
    assert out["_receipt"] == {"catalog_id": 3}


@pytest.mark.parametrize(
    "data",
    [
        # Самый длинный список пуст: ветка «offset за концом» отдавала всё целиком.
        {"id": 5, "body": "x" * 5000, "attachments": []},
        # Вес вне списка: окно в один элемент всё равно больше лимита.
        {"id": 5, "body": "x" * 5000, "members": [{"i": i} for i in range(5)]},
    ],
)
def test_write_whose_window_does_not_fit_gets_a_receipt(data):
    """Ответ больше лимита клиент может отбросить, и модель повторит запись:
    у пишущего вызова вместо него приходит квитанция."""
    out = server._adapt(FakeResponse(data), 0, 1024, write=True)
    assert out["_receipt"] == {"id": 5}
    assert "не повторяйте" in out["_pagination"]["note"]
    assert server._encoded_size(out) <= 1024


def test_list_bearing_write_over_the_limit_offers_no_next_offset():
    """next_offset у create_catalog — приглашение вызвать его снова, то есть
    создать второй справочник. Окно отдаётся, но продолжения нет."""
    items = [{"id": i, "v": "x" * 200} for i in range(20)]
    client = FakeClient(result=FakeResponse({"catalog_id": 3, "items": items}))
    out = _call(
        _build(client, Mode.NON_DESTRUCTIVE, max_bytes=1024),
        "create_catalog",
        {"request": {"name": "n"}},
    ).data
    page = out["_pagination"]
    assert page["returned"] < page["total"] == 20
    assert page["next_offset"] is None
    assert "не повторяйте" in page["note"]
    assert out["catalog_id"] == 3
    assert server._encoded_size(out) <= 1024
    assert len(client.calls) == 1


def test_sync_catalog_dry_run_pages_through_the_diff():
    """Дифф dry-run дальше первого окна достижим только через offset, а повтор
    dry-run ничего не пишет."""
    items = [{"id": i, "v": "x" * 200} for i in range(20)]
    mcp = _build(FakeClient(result=FakeResponse({"added": items})), Mode.NON_DESTRUCTIVE, max_bytes=1024)
    first = _call(mcp, "sync_catalog", {"entity_id": 1, "request": {"apply": False}}).data
    page = first["_pagination"]
    assert page["next_offset"] is not None
    assert "note" not in page
    second = _call(
        mcp, "sync_catalog", {"entity_id": 1, "request": {"apply": False}, "offset": page["next_offset"]}
    ).data
    assert second["_pagination"]["offset"] == page["next_offset"]


def test_sync_catalog_apply_is_a_write_and_offers_no_next_offset():
    """apply=true (только в full) повторно синхронизирует справочник, а между
    вызовами он мог измениться: это не повторяемый вызов."""
    items = [{"id": i, "v": "x" * 200} for i in range(20)]
    client = FakeClient(result=FakeResponse({"added": items}))
    mcp = _build(client, Mode.FULL, max_bytes=1024)
    out = _call(mcp, "sync_catalog", {"entity_id": 1, "request": {"apply": True}}).data
    assert out["_pagination"]["next_offset"] is None
    assert "не повторяйте" in out["_pagination"]["note"]
    with pytest.raises(ToolError, match="повторную запись"):
        _call(mcp, "sync_catalog", {"entity_id": 1, "request": {"apply": True}, "offset": 3})
    assert len(client.calls) == 1


@pytest.mark.filterwarnings("ignore:keys will default to True:DeprecationWarning")
def test_sync_catalog_dry_run_sends_apply_false():
    """Upstream ставит apply только для true, и без обхода тело было бы пустым:
    dry-run держался бы на неописанном умолчании Pyrus."""
    client = FakeClient()
    _call(
        _build(client, Mode.NON_DESTRUCTIVE),
        "sync_catalog",
        {"entity_id": 1, "request": {"apply": False, "items": []}},
    )
    obj = client.calls[0][1][1]
    assert json.loads(PyrusAPI().serialize_request(obj)) == {"apply": False}


def test_update_catalog_items_note_does_not_promise_the_rest_of_the_diff():
    """Дифф есть только в ответе записи: get_catalog отдаёт текущее состояние,
    а не added/updated/deleted."""
    items = [{"id": i, "v": "x" * 200} for i in range(20)]
    out = _call(
        _build(FakeClient(result=FakeResponse({"updated": items})), Mode.NON_DESTRUCTIVE, max_bytes=1024),
        "update_catalog_items",
        {"entity_id": 1, "request": {"upsert": []}},
    ).data
    note = out["_pagination"]["note"]
    assert "недоступны" in note
    assert "читающие инструменты" in note


def test_listless_read_over_the_limit_still_says_narrow_the_query():
    """Признак записи выводит _writes (Level и Spec.dry_run): чтение не должно его получить."""
    client = FakeClient(result=FakeResponse({"task": {"id": 1, "text": "x" * 5000}}))
    with pytest.raises(ToolError, match="Сузьте") as exc:
        _call(_build(client, Mode.READ_ONLY, max_bytes=1024), "get_task", {"entity_id": 1})
    assert "Вызов выполнен" not in str(exc.value)


def test_upload_file_over_the_limit_says_the_write_already_happened(tmp_path):
    """upload_file регистрируется руками, мимо _register: признак записи там ставится явно."""
    (tmp_path / "ok.txt").write_text("x")
    client = FakeClient(result=FakeResponse({"guid": "g1", "blob": "x" * 5000}))
    out = _call(
        _build(client, Mode.FULL, tmp_path.resolve(), max_bytes=1024),
        "upload_file",
        {"file_path": "ok.txt"},
    ).data
    assert "Вызов выполнен" in out["_pagination"]["note"]
    assert out["_receipt"] == {"guid": "g1"}


def test_set_avatar_over_the_limit_says_the_write_already_happened():
    """set_avatar, как и upload_file, регистрируется мимо _register."""
    client = FakeClient(result=FakeResponse({"id": 1, "blob": "x" * 5000}))
    out = _call(
        _build(client, Mode.FULL, max_bytes=1024),
        "set_avatar",
        {"member_id": 1, "file_guid": "g"},
    ).data
    assert "Вызов выполнен" in out["_pagination"]["note"]
    assert out["_receipt"] == {"id": 1}
    assert len(client.calls) == 1


def test_destructive_tool_is_a_write_for_offset_and_receipt():
    """DESTRUCTIVE — тоже запись: повтор удаления ради окна недопустим."""
    client = FakeClient(result=FakeResponse({"id": 9, "blob": "x" * 5000}))
    mcp = _build(client, Mode.FULL, max_bytes=1024)
    with pytest.raises(ToolError, match="повторную запись"):
        _call(mcp, "delete_knowledge_base_entity", {"entity_id": 9, "offset": 2})
    assert client.calls == []
    out = _call(mcp, "delete_knowledge_base_entity", {"entity_id": 9}).data
    assert out["_receipt"] == {"id": 9}
    assert len(client.calls) == 1


class _Raising(FakeClient):
    """Вызов падает с заданным исключением вместо ответа."""

    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            raise self.exc

        return call


def _DroppedConnection():
    """Pyrus мог успеть записать, а ответ до клиента не дошёл."""
    return _Raising(requests.ConnectionError("connection reset"))


def test_write_that_raises_says_the_outcome_is_unknown():
    """Обычная ошибка после отправленного POST читается как «не получилось»
    и ведёт к дублю: модель должна сначала проверить состояние чтением."""
    client = _DroppedConnection()
    with pytest.raises(ToolError, match="Исход вызова неизвестен"):
        _call(
            _build(client, Mode.NON_DESTRUCTIVE),
            "comment_task",
            {"entity_id": 1, "request": {"formatted_text": "<b>x</b>"}},
        )
    assert len(client.calls) == 1


def test_write_refused_before_sending_is_not_an_unknown_outcome():
    """pyrus-api проверяет аргументы до запроса (тип entity_id, размер файла):
    записи точно не было, и «проверьте состояние» отправило бы модель в никуда."""
    client = _Raising(TypeError("entity_id must be an instance of str"))
    with pytest.raises(ToolError) as exc:
        _call(
            _build(client, Mode.NON_DESTRUCTIVE),
            "comment_task",
            {"entity_id": 1, "request": {"formatted_text": "<b>x</b>"}},
        )
    assert "Исход вызова неизвестен" not in str(exc.value)


def test_read_that_raises_does_not_claim_a_write():
    client = _DroppedConnection()
    with pytest.raises(ToolError) as exc:
        _call(_build(client, Mode.READ_ONLY), "get_task", {"entity_id": 1})
    assert "Исход вызова неизвестен" not in str(exc.value)


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
    assert "formatted_text" in str(exc.value)


_FT = '<b>жирный</b><br><span data-personid="1" data-type="user-mention">Имя</span>'


# pyrus-api зовёт jsonpickle.encode без keys=, а jsonpickle 4 на это
# предупреждает. Фильтр терпит предупреждение, но не требует его: pytest.warns
# упал бы с DID NOT WARN, как только upstream передаст keys= сам.
@pytest.mark.filterwarnings("ignore:keys will default to True:DeprecationWarning")
@pytest.mark.parametrize(
    "method", ["create_task", "comment_task", "create_announcement", "comment_announcement"]
)
def test_formatted_text_reaches_the_request_body(method):
    """Upstream-классы formatted_text не знают: ключ ставится после конструктора
    и должен доехать до тела запроса тем же serialize_request, что в production."""
    obj = server._build_request(_spec(method), {"formatted_text": _FT})
    assert vars(obj)["formatted_text"] == _FT
    body = json.loads(PyrusAPI().serialize_request(obj))
    assert body["formatted_text"] == _FT


@pytest.mark.parametrize("empty", ["", None])
def test_empty_formatted_text_is_not_set(empty):
    obj = server._build_request(_spec("comment_task"), {"text": "ок", "formatted_text": empty})
    assert obj.text == "ок"
    assert "formatted_text" not in vars(obj)


def test_bpmn_keys_of_a_comment_reach_the_request_object():
    payload = {
        "bpmn_node_id": "node-1",
        "bpmn_resume_processing": True,
        "bpmn_token_moves": [{"from": "a", "to": "b"}],
    }
    obj = server._build_request(_spec("comment_task"), payload)
    assert {k: vars(obj)[k] for k in payload} == payload


def test_explicit_false_bpmn_value_is_sent_not_dropped():
    """Значение по умолчанию у Pyrus неизвестно, поэтому явный false уходит как есть."""
    payload = {"bpmn_resume_processing": False, "bpmn_token_moves": []}
    obj = server._build_request(_spec("comment_task"), payload)
    assert {k: vars(obj)[k] for k in payload} == payload


def test_non_bool_bpmn_resume_processing_is_refused_before_the_write():
    """Строка "false" ушла бы в Pyrus как есть; как он её прочтёт, неизвестно."""
    client = FakeClient()
    with pytest.raises(ToolError, match="bpmn_resume_processing"):
        _call(
            _build(client, Mode.NON_DESTRUCTIVE),
            "comment_task",
            {"entity_id": 1, "request": {"bpmn_resume_processing": "false"}},
        )
    assert client.calls == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("bpmn_node_id", {"id": 1}),
        ("bpmn_node_id", True),
        ("bpmn_token_moves", "a->b"),
        ("bpmn_token_moves", {"from": "a"}),
    ],
)
def test_wrong_typed_bpmn_value_is_refused_before_the_write(key, value):
    """Объект в bpmn_node_id или строка в bpmn_token_moves ушли бы в Pyrus как есть."""
    client = FakeClient()
    with pytest.raises(ToolError, match=key):
        _call(
            _build(client, Mode.NON_DESTRUCTIVE),
            "comment_task",
            {"entity_id": 1, "request": {key: value}},
        )
    assert client.calls == []


@pytest.mark.parametrize("value", [["<b>x</b>"], [], 0, {}])
def test_non_string_formatted_text_is_refused_before_the_write(value):
    """Пустой список не должен молча выпасть: тогда ушёл бы пустой комментарий."""
    client = FakeClient()
    with pytest.raises(ToolError, match="строкой с HTML"):
        _call(
            _build(client, Mode.NON_DESTRUCTIVE),
            "comment_task",
            {"entity_id": 1, "request": {"formatted_text": value}},
        )
    assert client.calls == []


def test_comment_task_description_teaches_formatting_and_mentions():
    text = server._describe(_spec("comment_task"), Mode.NON_DESTRUCTIVE)
    assert "formatted_text" in text
    assert 'data-type="user-mention"' in text
    assert "<br>" in text and "&amp;" in text
    assert "в ответе этого же вызова" in text
    assert "get_contacts / get_members" in text


def test_specs_without_extra_keys_get_no_formatting_hint():
    assert "formatted_text" not in server._describe(_spec("create_role"), Mode.NON_DESTRUCTIVE)


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
    assert "read-only" in (mcp.instructions or "")


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


def test_formatted_text_reaches_pyrus_through_the_tool():
    client = FakeClient()
    _call(
        _build(client, Mode.NON_DESTRUCTIVE),
        "comment_task",
        {"entity_id": 42, "request": {"formatted_text": "<b>ок</b>"}},
    )
    request = client.calls[0][1][1]
    assert request.formatted_text == "<b>ок</b>"
    assert "text" not in vars(request)


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
    server._register(mcp, cast(PyrusAPI, client), spec, Mode.NON_DESTRUCTIVE, 65536)
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


def test_positive_offset_on_a_write_is_refused_before_the_write():
    """Окно пишущего вызова со сдвигом — это вторая задача ради того же ответа.
    Отказ до вызова ничего не теряет: записи ещё не было."""
    client = FakeClient(result=FakeResponse({"task": {"id": 42}}))
    with pytest.raises(ToolError, match="повторную запись"):
        _call(
            _build(client, Mode.NON_DESTRUCTIVE),
            "create_task",
            {"request": {"text": "x"}, "offset": 5},
        )
    assert client.calls == []


def test_positive_offset_after_a_write_still_returns_the_result():
    """_adapt зовут и мимо инструментов. Там запись уже состоялась, и отказ
    потерял бы id созданной задачи."""
    out = server._adapt(FakeResponse({"task": {"id": 42}}), 5, 1024, write=True)
    assert out["task"] == {"id": 42}
    assert out["_pagination"]["field"] is None


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
