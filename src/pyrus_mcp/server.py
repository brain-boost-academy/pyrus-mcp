"""Сборка FastMCP-сервера: интроспекция ключей, диспетчер, адаптер результата, main().

stdout занят MCP-протоколом. Всё, что нужно сказать человеку, идёт в stderr.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, overload

import jsonpickle
import requests
import pyrus.models.entities as entities
import pyrus.models.requests as pyrus_requests
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pyrus.client import PyrusAPI

from .catalog import CATALOG, Level, Mode, Shape, Spec, visible

DEFAULT_MAX_RESULT_BYTES = 256 * 1024

# CreateMemberRequest / UpdateMemberRequest объявлены как **kwargs над
# entities.Person, поэтому inspect.signature по ним не даёт ни одного ключа,
# а Person молча выбрасывает незнакомые ключи — опечатка не вызвала бы ошибки
# ни здесь, ни на стороне Pyrus. Ключи вынимаются из тела Person.__init__.
# ponytail: разбор исходника регуляркой. Если апстрим перепишет Person на
# явные параметры, вернётся пустой набор и валидация просто отключится —
# см. ветку «ключи не валидируются» в _describe.
_PERSON_KEYS: tuple[str, ...] = tuple(
    re.findall(r"""if ['"](\w+)['"] in kwargs""", inspect.getsource(entities.Person.__init__))
)


def _keys(func: Any, skip_first_positional: bool = False) -> tuple[list[str], list[str]]:
    """(валидные ключи, обязательные ключи) для callable."""
    params = [
        p
        for name, p in inspect.signature(func).parameters.items()
        if name != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    if skip_first_positional and params:
        params = params[1:]
    if not params and any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(func).parameters.values()
    ):
        return list(_PERSON_KEYS), []
    return [p.name for p in params], [
        p.name for p in params if p.default is inspect.Parameter.empty
    ]


def _spec_keys(spec: Spec) -> tuple[list[str], list[str]]:
    if spec.request:
        keys, required = _keys(getattr(pyrus_requests, spec.request).__init__)
        return keys + list(spec.extra_keys), required
    return _keys(getattr(PyrusAPI, spec.method), skip_first_positional=spec.shape is Shape.BY_ID)


# Разметка упоминания официально не описана: снята с живых комментариев
# через get_task. Плоский text упоминание не создаёт, в нём остаётся только имя.
_FORMATTED_TEXT_HINT = (
    "Форматирование и упоминания задаются только через formatted_text (HTML; размечаются "
    "только перечисленные теги). Передавайте formatted_text; text для комментариев не нужен; "
    "если Pyrus отклонит сочетание полей, ошибка вернётся целиком. "
    'Теги: <b> <i> <s> <code> <q> <br>, <div data-type="heading">, '
    '<mark data-color="red|yellow|green|blue">, <ul>/<ol><li>, <a href>. '
    "Перевод строки — <br>; символы & < > в тексте экранируйте как &amp; &lt; &gt;. "
    'Упоминание: <span data-personid="ID" data-type="user-mention">Имя</span>, '
    "ID из get_contacts / get_members. Pyrus возвращает распознанные упоминания в поле "
    "mentions комментария в ответе этого же вызова, повторный вызов не нужен. "
    "Если ответ не влез в лимит, вместо него придёт _receipt: id задачи или объявления "
    "и id + mentions последнего комментария (не обязательно вашего: например, при "
    "edit_comment_id у comment_task). "
    "Запись при этом уже выполнена: не повторяйте вызов."
)


def _describe(spec: Spec, mode: Mode) -> str:
    """Описание инструмента. Ключи payload снимаются с upstream на импорте,
    поэтому обновление pyrus-api не рассинхронизирует их. Исключение —
    Spec.extra_keys: они заданы руками, дрейф ловит тест в test_catalog.py."""
    doc = re.split(
        r"\n\s*(?:Args|Returns):", inspect.getdoc(getattr(PyrusAPI, spec.method)) or ""
    )[0].strip()
    keys, required = _spec_keys(spec)
    field = "request" if spec.request else "options"
    parts = [doc or spec.method]
    if keys:
        parts.append(f"Ключи {field}: {', '.join(sorted(keys))}.")
        if required:
            parts.append(f"Обязательные: {', '.join(required)}.")
        if spec.request and not required and set(keys) == set(_PERSON_KEYS):
            parts.append("Незнакомые ключи молча игнорируются Pyrus, а не отклоняются.")
    elif spec.request:
        parts.append(f"Ключи {field} не валидируются: upstream принимает произвольные kwargs.")
    if "formatted_text" in spec.extra_keys:
        parts.append(_FORMATTED_TEXT_HINT)
    if spec.guard and mode is not Mode.FULL:
        parts.append(f"Режим {mode}: часть полей заблокирована, вызов вернёт ошибку с названием поля.")
    return " ".join(parts)


# --- адаптер результата ----------------------------------------------------


def _encoded_size(data: Any) -> int:
    return len(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def _to_plain(result: Any) -> Any:
    """Ответ pyrus → обычный JSON. BaseResponse хранит сырой ответ API в
    original_response, поэтому в типичном случае jsonpickle не нужен вовсе."""
    raw = getattr(result, "original_response", None)
    if isinstance(raw, dict) and raw:
        return raw
    encoded: Any = jsonpickle.encode(result, unpicklable=False, keys=True)
    return json.loads(encoded)


def _longest_list_key(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    lists = [(len(v), k) for k, v in data.items() if isinstance(v, list)]
    return max(lists)[1] if lists else None


def _check_offset(offset: int, *, write: bool = False) -> None:
    """Проверки offset, которые можно сделать до вызова Pyrus. Зовётся дважды:
    из инструмента — до диспетча, из _paginate — как инвариант самой
    пагинации, потому что _adapt зовут и напрямую. Там запись уже состоялась,
    поэтому _paginate проверяет только знак."""
    # Отрицательное значение молча отдало бы хвост списка под видом окна
    # с начала, а next_offset отправил бы модель на второй проход.
    if offset < 0:
        raise ToolError(f"offset не может быть отрицательным, получено {offset}.")
    # Окно пишущего вызова со сдвигом — это вторая запись ради того же ответа:
    # второй комментарий или второй справочник.
    if write and offset != 0:
        raise ToolError(
            f"offset у пишущего вызова неприменим, получено {offset}: следующее окно "
            "означало бы повторную запись. Вызовите без offset."
        )


def _writes(spec: Spec, payload: dict) -> bool:
    """Меняет ли вызов данные. Уровень спека этого не решает: dry-run
    sync_catalog живёт на WRITE, но ничего не пишет."""
    return spec.level is not Level.READ and not (spec.dry_run and spec.dry_run(payload))


def _write_receipt(data: Any) -> dict:
    """Что модели нужно знать о состоявшейся записи, если весь ответ не влез:
    id задачи или объявления и id + mentions последнего комментария. Без этого
    обещание подсказки «mentions в ответе этого же вызова» ложно ровно на
    больших задачах. Что новый комментарий в ответе последний, вживую не
    проверялось, поэтому ключ называется last_comment, а не new_comment."""
    receipt: dict = {}
    if not isinstance(data, dict):
        return receipt
    # upload_file отвечает guid на верхнем уровне, и без него файл не прикрепить.
    # *_id — на случай ответа с идентификатором под своим именем (catalog_id):
    # квитанция без id после «не повторяйте вызов» теряет созданную запись.
    for key, value in data.items():
        if (key in ("id", "guid") or key.endswith("_id")) and isinstance(value, (int, str)):
            receipt[key] = value
    for entity in ("task", "announcement"):
        obj = data.get(entity)
        if not isinstance(obj, dict) or "id" not in obj:
            continue
        receipt[f"{entity}_id"] = obj["id"]
        comments = obj.get("comments")
        if isinstance(comments, list) and comments and isinstance(comments[-1], dict):
            last = comments[-1]
            receipt["last_comment"] = {k: last[k] for k in ("id", "mentions") if k in last}
    return receipt


def _receipt_result(data: Any, offset: int, size: int, max_bytes: int) -> dict:
    """Ответ пишущего вызова, который в лимит не влез. Ошибка (isError) или
    ответ больше лимита читались бы клиентом или моделью как «не получилось»
    и вели к дублю записи, поэтому отдаётся результат: квитанция вместо ответа."""
    return {
        "_receipt": _write_receipt(data),
        "_pagination": {
            "field": None,
            "offset": offset,
            "returned": None,
            "total": None,
            "next_offset": None,
            "note": "Вызов выполнен, изменение уже применено; не повторяйте вызов. "
            f"Ответ {size} байт не влез в лимит {max_bytes}, и нарезать его так, чтобы "
            "влез, нельзя, поэтому вместо него — _receipt. Чтобы видеть такие ответы "
            "целиком, поднимите PYRUS_MCP_MAX_RESULT_BYTES.",
        },
    }


def _paginate(data: Any, offset: int, max_bytes: int, *, write: bool = False) -> Any:
    """Режет самый длинный список верхнего уровня так, чтобы результат влез в
    лимит. Никогда не усекает молча: окно всегда подписано total и next_offset.
    У пишущего вызова (write) продолжения нет: next_offset всегда None, а
    остаток назван в note. Если ответ записи не нарезать в лимит, вместо него
    отдаётся _receipt (_receipt_result)."""
    _check_offset(offset)

    size = _encoded_size(data)
    if offset == 0 and size <= max_bytes:
        return data

    key = _longest_list_key(data)
    if key is None:
        if size > max_bytes:
            # Пишущий вызов уже состоялся: comment_task на большой задаче
            # отвечает всей задачей без списка верхнего уровня.
            if write:
                return _receipt_result(data, offset, size, max_bytes)
            raise ToolError(
                f"Результат {size} байт превышает лимит {max_bytes} и не содержит "
                "списка, который можно нарезать. Сузьте запрос фильтрами или поднимите "
                "PYRUS_MCP_MAX_RESULT_BYTES."
            )
        # Сюда попадают только с offset != 0, и вызов уже состоялся. Если он был
        # пишущим, выбросить его результат ради претензии к аргументу — значит
        # потерять id созданной записи; прежний текст «повторите с offset=0»
        # прямо приказывал модели повторить незаидемпотентную запись.
        return {
            **data,
            "_pagination": {
                "field": None,
                "offset": offset,
                "returned": None,
                "total": None,
                "next_offset": None,
                "note": "В ответе нет списка верхнего уровня, offset неприменим. "
                "Вызов выполнен, результат возвращён целиком; повторять его не нужно.",
            },
        }

    items = data[key]
    total = len(items)
    if offset >= total:
        out = {
            **data,
            key: [],
            "_pagination": {
                "field": key,
                "offset": offset,
                "returned": 0,
                "total": total,
                "next_offset": None,
            },
        }
        # Самый длинный список пуст, а вес лежит рядом с ним (тело статьи у
        # create_knowledge_base_entity): резать нечего.
        if write and _encoded_size(out) > max_bytes:
            return _receipt_result(data, offset, size, max_bytes)
        return out

    count = total - offset
    while True:
        window = items[offset : offset + count]
        more = offset + len(window) < total
        page: dict = {
            "field": key,
            "offset": offset,
            "returned": len(window),
            "total": total,
            "next_offset": offset + len(window) if more and not write else None,
        }
        # Следующее окно пишущего вызова — это повтор записи (второй справочник
        # у create_catalog), поэтому продолжения нет, а остаток честно назван.
        # Читающий инструмент его не заменяет: дифф update_catalog_items есть
        # только в этом ответе, а get_catalog отдаёт текущее состояние.
        # Пометка ставится до замера: она тоже занимает место в лимите.
        if write and more:
            page["note"] = (
                "Вызов выполнен, изменение уже применено; не повторяйте вызов: у пишущего "
                f"инструмента следующее окно — это повторная запись. Показаны {len(window)} "
                f"из {total} элементов {key}, остальные элементы этого ответа недоступны. "
                "Целиком такие ответы видны при большем PYRUS_MCP_MAX_RESULT_BYTES; текущее "
                "состояние данных отдают читающие инструменты."
            )
        out = {**data, key: window, "_pagination": page}
        if _encoded_size(out) <= max_bytes or count <= 1:
            break
        count //= 2
    # Окно в один элемент может не влезть, если вес вне списка. Для чтения это
    # открытый TODO; запись повторять нельзя, поэтому — квитанция.
    if write and _encoded_size(out) > max_bytes:
        return _receipt_result(data, offset, size, max_bytes)
    return out


def _dispatch(call: Callable[[], Any], write: bool) -> Any:
    """Вызов Pyrus. У пишущего вызова сетевая ошибка или ответ, который не
    разобрался как JSON (502-страница), не значат, что записи не было: обычная
    ошибка читалась бы как «не получилось» и вела к дублю, поэтому модель
    получает «исход неизвестен». Прочие исключения pyrus-api бросает до
    запроса (тип entity_id, размер файла): записи точно не было, и они
    уходят как есть. Зависший запрос сюда не доходит: таймаута у upstream нет."""
    if not write:
        return call()
    try:
        return call()
    except (requests.RequestException, ValueError) as exc:
        raise ToolError(
            f"Исход вызова неизвестен ({type(exc).__name__}): изменение могло уже "
            "примениться. Прежде чем повторять, проверьте текущее состояние читающим "
            "инструментом."
        ) from exc


def _adapt(result: Any, offset: int, max_bytes: int, *, write: bool = False) -> Any:
    code = getattr(result, "error_code", None)
    if code:
        raise ToolError(f"Pyrus вернул ошибку {code}: {getattr(result, 'error', '') or '—'}")
    return _paginate(_to_plain(result), offset, max_bytes, write=write)


# --- диспетчер -------------------------------------------------------------


def _check_guard(spec: Spec, payload: dict, mode: Mode) -> None:
    if spec.guard is None or mode is Mode.FULL:
        return
    if (field := spec.guard(payload)) is not None:
        raise ToolError(
            f"{spec.method}: поле {field} недоступно в режиме {mode}. "
            "Перезапустите сервер с --mode full."
        )


def _build_request(spec: Spec, payload: dict) -> Any:
    # Несущая проверка, а не документация невозможного: при spec.request is None
    # _spec_keys падает в сигнатуру самого метода, поэтому строка REQ/ID_REQ без
    # request-класса регистрируется штатно и взрывается только здесь, на вызове.
    # Каталог проверяется целиком в test_every_req_shape_names_a_request_class;
    # этот assert — второй рубеж, и под python -O его нет.
    assert spec.request, spec.method
    cls = getattr(pyrus_requests, spec.request)
    base = {k: v for k, v in payload.items() if k not in spec.extra_keys}
    # Не ставятся только None и пустая строка, как `if text:` в upstream.
    # Прочие ложные значения уходят как есть: явный false у bpmn-ключа значим,
    # а пустой список в formatted_text должен дойти до проверки типа ниже.
    extra = {k: v for k, v in payload.items() if k in spec.extra_keys and v not in (None, "")}
    if not isinstance(extra.get("formatted_text", ""), str):
        raise ToolError(f"{spec.method}: formatted_text должен быть строкой с HTML.")
    # Allowlist, как у guard'ов: строка "false" ушла бы в Pyrus как есть.
    if not isinstance(extra.get("bpmn_resume_processing", False), bool):
        raise ToolError(f"{spec.method}: bpmn_resume_processing должен быть true или false.")
    node = extra.get("bpmn_node_id", "")
    if isinstance(node, bool) or not isinstance(node, (str, int)):
        raise ToolError(f"{spec.method}: bpmn_node_id должен быть строкой или числом.")
    if not isinstance(extra.get("bpmn_token_moves", []), list):
        raise ToolError(f"{spec.method}: bpmn_token_moves должен быть списком.")
    try:
        obj = cls(**base)
    except TypeError as exc:
        keys, required = _spec_keys(spec)
        raise ToolError(
            f"{spec.method}: неверный состав request ({exc}). "
            f"Валидные ключи: {', '.join(sorted(keys)) or '—'}. "
            f"Обязательные: {', '.join(required) or 'нет'}."
        ) from exc
    for key, value in extra.items():
        setattr(obj, key, value)
    for key in spec.keep_false:
        if base.get(key) is False:
            setattr(obj, key, False)
    return obj


def _call_plain(client: PyrusAPI, spec: Spec, args: list, options: dict) -> Any:
    try:
        return getattr(client, spec.method)(*args, **options)
    except TypeError as exc:
        keys, _ = _spec_keys(spec)
        raise ToolError(
            f"{spec.method}: неверный состав options ({exc}). "
            f"Валидные ключи: {', '.join(sorted(keys)) or 'нет параметров'}."
        ) from exc


# --- регистрация -----------------------------------------------------------


def _register(mcp: FastMCP, client: PyrusAPI, spec: Spec, mode: Mode, max_bytes: int) -> None:
    """Одна строка таблицы → один инструмент MCP. Форма подписи выбирается по
    spec.shape и наличию ключей у request-класса.

    Обновлять рисунок вместе с любой новой Shape: устаревший хуже отсутствующего.

        Shape (ключи)      | подпись инструмента
        -------------------+------------------------------------
        PLAIN  (нет ключей)| tool(offset)
        PLAIN              | tool(options, offset)
        BY_ID  (нет ключей)| tool(entity_id, offset)
        BY_ID              | tool(entity_id, options, offset)
        REQ                | tool(request, offset)
        ID_REQ (required)  | tool(entity_id, request, offset)
        ID_REQ             | tool(entity_id, request?, offset)
                             ^ offset есть у всех, но у пишущего вызова
                               (_writes) отклоняется любой, кроме 0; листать
                               можно только чтение и dry-run sync_catalog

        вызов: args -> _check_offset -> _check_guard -> _build_request -> _dispatch -> PyrusAPI
                          | offset < 0,    | mode != FULL  | неверный ключ | сеть / не JSON
                          | offset != 0 у  |               | или тип extra | у пишущего
                          | пишущего       |               |               |
                          +-> ToolError    +-> ToolError   +-> ToolError   +-> ToolError
                          ^ первые три до вызова: после      (исход неизвестен)
                            него отказ означал бы «запись    |
                            состоялась, а модель получила    v
                            отказ»
                                     _adapt -> _to_plain -> _paginate -> окно
                                        | error_code              |   + _pagination
                                        +-> ToolError             +-> _receipt (запись
                                                                      не влезла в лимит)

    ONE_OFF сюда не приходит: три ручных инструмента живут в
    _register_one_offs.
    """
    method = getattr(client, spec.method)
    description = _describe(spec, mode)

    keys, required = _spec_keys(spec)

    def check(payload: dict, offset: int) -> bool:
        """Всё, что отклоняется до вызова Pyrus. Возвращает, пишущий ли вызов."""
        write = _writes(spec, payload)
        _check_offset(offset, write=write)
        _check_guard(spec, payload, mode)
        return write

    def adapt(call: Callable[[], Any], offset: int, write: bool) -> Any:
        return _adapt(_dispatch(call, write), offset, max_bytes, write=write)

    match spec.shape:
        case Shape.PLAIN if not keys:

            def tool(offset: int = 0) -> Any:  # pyright: ignore[reportRedeclaration]
                write = check({}, offset)
                return adapt(lambda: method(), offset, write)

        case Shape.PLAIN:

            def tool(options: dict | None = None, offset: int = 0) -> Any:  # pyright: ignore[reportRedeclaration]
                payload = options or {}
                write = check(payload, offset)
                return adapt(lambda: _call_plain(client, spec, [], payload), offset, write)

        case Shape.BY_ID if not keys:

            def tool(entity_id: int, offset: int = 0) -> Any:  # pyright: ignore[reportRedeclaration]
                write = check({}, offset)
                return adapt(lambda: method(entity_id), offset, write)

        case Shape.BY_ID:

            def tool(entity_id: int, options: dict | None = None, offset: int = 0) -> Any:  # pyright: ignore[reportRedeclaration]
                payload = options or {}
                write = check(payload, offset)
                return adapt(lambda: _call_plain(client, spec, [entity_id], payload), offset, write)

        case Shape.REQ:

            def tool(request: dict, offset: int = 0) -> Any:  # pyright: ignore[reportRedeclaration]
                write = check(request, offset)
                req = _build_request(spec, request)
                return adapt(lambda: method(req), offset, write)

        # Схема не врёт: если у request-класса есть обязательные поля,
        # параметр request обязателен и в сигнатуре инструмента.
        case Shape.ID_REQ if required:

            def tool(entity_id: int, request: dict, offset: int = 0) -> Any:  # pyright: ignore[reportRedeclaration]
                write = check(request, offset)
                req = _build_request(spec, request)
                return adapt(lambda: method(entity_id, req), offset, write)

        case Shape.ID_REQ:

            def tool(entity_id: int, request: dict | None = None, offset: int = 0) -> Any:  # pyright: ignore[reportRedeclaration]
                payload = request or {}
                write = check(payload, offset)
                req = _build_request(spec, payload)
                return adapt(lambda: method(entity_id, req), offset, write)

        case _:  # pragma: no cover — ONE_OFF регистрируется отдельно
            raise AssertionError(spec.shape)

    if spec.shape in (Shape.BY_ID, Shape.ID_REQ):
        description = f"Первый аргумент entity_id — это {spec.id_arg}. {description}"

    mcp.tool(tool, name=spec.method, description=description)


def _register_one_offs(
    mcp: FastMCP, client: PyrusAPI, specs: set[str], file_root: Path | None, max_bytes: int
) -> None:
    if "download_file" in specs:

        def download_file(file_id: int) -> dict:
            """Скачать вложение и вернуть его содержимое инлайн. На диск ничего
            не пишется. Текст возвращается как есть, бинарные данные — base64."""
            # BaseResponse не объявляет raw_file/filename — их приносит
            # DownloadResponse, и то как = None.
            result: Any = client.download_file(file_id)
            if code := getattr(result, "error_code", None):
                raise ToolError(f"Pyrus вернул ошибку {code}")
            raw: bytes = result.raw_file
            if len(raw) > max_bytes:
                raise ToolError(
                    f"Файл {len(raw)} байт превышает лимит {max_bytes}. "
                    "Поднимите PYRUS_MCP_MAX_RESULT_BYTES."
                )
            try:
                return {"filename": result.filename, "encoding": "utf-8", "content": raw.decode()}
            except UnicodeDecodeError:
                return {
                    "filename": result.filename,
                    "encoding": "base64",
                    "content": base64.b64encode(raw).decode(),
                }

        mcp.tool(download_file)

    if "upload_file" in specs and file_root is not None:

        def upload_file(file_path: str) -> dict:
            """Загрузить локальный файл в Pyrus и получить guid для вложения.
            Путь обязан лежать внутри PYRUS_MCP_FILE_ROOT."""
            candidate = Path(file_path)
            target = (file_root / candidate).resolve()
            if not target.is_relative_to(file_root):
                raise ToolError(
                    f"Путь вне разрешённого корня {file_root}. Символические ссылки наружу "
                    "и переходы через '..' отклоняются."
                )
            if not target.is_file():
                raise ToolError(f"Файл не найден: {target}")
            result = _dispatch(lambda: client.upload_file(str(target)), write=True)
            return _adapt(result, 0, max_bytes, write=True)

        mcp.tool(upload_file)

    if "set_avatar" in specs:

        def set_avatar(member_id: int, file_guid: str, external_avatar_id: str | None = None) -> Any:
            """Назначить аватар пользователю. file_guid — идентификатор,
            полученный от upload_file, а не путь на диске."""
            result = _dispatch(
                lambda: client.set_avatar(member_id, file_guid, external_avatar_id), write=True
            )
            return _adapt(result, 0, max_bytes, write=True)

        mcp.tool(set_avatar)


def build(client: PyrusAPI, mode: Mode, file_root: Path | None, max_bytes: int) -> FastMCP:
    specs = visible(mode, file_root=file_root is not None)
    mcp = FastMCP(
        name="pyrus-mcp",
        instructions=(
            f"Доступ к Pyrus в режиме «{mode}». "
            f"Зарегистрировано инструментов: {len(specs)} из {len(CATALOG)}. "
            "Недоступные в этом режиме методы отсутствуют в списке инструментов — "
            "их нельзя вызвать, и пытаться не нужно. "
            "Смена режима возможна только перезапуском сервера с другим --mode."
        ),
    )
    one_offs = {s.method for s in specs if s.shape is Shape.ONE_OFF}
    for spec in specs:
        if spec.shape is not Shape.ONE_OFF:
            _register(mcp, client, spec, mode, max_bytes)
    _register_one_offs(mcp, client, one_offs, file_root, max_bytes)
    return mcp


# --- запуск ----------------------------------------------------------------


def _make_client(**kwargs: Any) -> PyrusAPI:
    """Отдельная точка создания клиента: тесты подменяют её, а интроспекция
    ключей продолжает читать настоящий класс PyrusAPI."""
    return PyrusAPI(**kwargs)


def _die(message: str) -> NoReturn:
    print(f"pyrus-mcp: {message}", file=sys.stderr)
    raise SystemExit(2)


@overload
def _int_env(name: str, default: int, *, positive: bool = False) -> int: ...
@overload
def _int_env(name: str, default: None = None, *, positive: bool = False) -> int | None: ...
def _int_env(name: str, default: int | None = None, *, positive: bool = False) -> int | None:
    """Числовая переменная окружения. Мусор в ней — такая же ошибка конфигурации,
    как отсутствующий логин, и должна выходить так же, а не трейсбеком."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        _die(f"{name} должен быть целым числом, получено {raw!r}")
    if positive and value <= 0:
        _die(f"{name} должен быть положительным, получено {value}")
    return value


def resolve_mode(argv: list[str] | None = None) -> Mode:
    parser = argparse.ArgumentParser(prog="pyrus-mcp")
    parser.add_argument("--mode", default=None)
    flag = parser.parse_args(argv).mode
    raw = flag or os.environ.get("PYRUS_MCP_MODE") or Mode.NON_DESTRUCTIVE.value
    try:
        return Mode(raw)
    except ValueError:
        _die(f"неизвестный режим {raw!r}. Допустимые: {', '.join(m.value for m in Mode)}")


def main(argv: list[str] | None = None) -> None:
    mode = resolve_mode(argv)

    login = os.environ.get("PYRUS_LOGIN")
    key = os.environ.get("PYRUS_SECURITY_KEY")
    if not login or not key:
        _die("нужны переменные окружения PYRUS_LOGIN и PYRUS_SECURITY_KEY")

    root_raw = os.environ.get("PYRUS_MCP_FILE_ROOT")
    file_root = Path(root_raw).resolve() if root_raw else None
    if file_root is not None and not file_root.is_dir():
        _die(f"PYRUS_MCP_FILE_ROOT указывает не на каталог: {file_root}")

    max_bytes = _int_env("PYRUS_MCP_MAX_RESULT_BYTES", DEFAULT_MAX_RESULT_BYTES, positive=True)

    # ponytail: один общий PyrusAPI без лока. Ре-авторизация по 401 мутирует
    # self.access_token на месте, а FastMCP уводит синхронные инструменты
    # в threadpool — при одном stdio-клиенте гонка практически невозможна.
    # Ставить лок, если появится HTTP-транспорт или несколько клиентов.
    client = _make_client(
        login=login,
        security_key=key,
        access_token=os.environ.get("PYRUS_ACCESS_TOKEN"),
        person_id=_int_env("PYRUS_PERSON_ID"),
    )

    # auth() возвращает объект с error_code, а не бросает исключение:
    # try/except вокруг него не поймал бы ничего, и сервер стартовал бы
    # с мёртвыми ключами.
    auth = client.auth()
    if code := getattr(auth, "error_code", None):
        _die(f"авторизация не прошла: {code}")

    mcp = build(client, mode, file_root, max_bytes)
    count = len(visible(mode, file_root=file_root is not None))
    print(f"pyrus-mcp: режим {mode}, инструментов {count}", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
