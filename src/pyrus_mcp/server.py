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
from pathlib import Path
from typing import Any

import jsonpickle
import pyrus.models.entities as entities
import pyrus.models.requests as pyrus_requests
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pyrus.client import PyrusAPI

from .catalog import CATALOG, Mode, Shape, Spec, visible

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
        return _keys(getattr(pyrus_requests, spec.request).__init__)
    return _keys(getattr(PyrusAPI, spec.method), skip_first_positional=spec.shape is Shape.BY_ID)


def _describe(spec: Spec, mode: Mode) -> str:
    """Описание инструмента. Ключи payload не хардкодятся — они снимаются с
    upstream на импорте, поэтому обновление pyrus-api не рассинхронизирует их."""
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
    return json.loads(jsonpickle.encode(result, unpicklable=False, keys=True))


def _longest_list_key(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    lists = [(len(v), k) for k, v in data.items() if isinstance(v, list)]
    return max(lists)[1] if lists else None


def _check_offset(offset: int) -> None:
    """Единственная проверка offset, которую можно сделать до вызова Pyrus.
    Зовётся дважды: из инструмента — до диспетча, из _paginate — как инвариант
    самой пагинации, потому что _adapt зовут и напрямую."""
    # Отрицательное значение молча отдало бы хвост списка под видом окна
    # с начала, а next_offset отправил бы модель на второй проход.
    if offset < 0:
        raise ToolError(f"offset не может быть отрицательным, получено {offset}.")


def _paginate(data: Any, offset: int, max_bytes: int) -> Any:
    """Режет самый длинный список верхнего уровня так, чтобы результат влез в
    лимит. Никогда не усекает молча: окно всегда подписано total и next_offset."""
    _check_offset(offset)

    size = _encoded_size(data)
    if offset == 0 and size <= max_bytes:
        return data

    key = _longest_list_key(data)
    if key is None:
        if size > max_bytes:
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
        return {
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

    count = total - offset
    while True:
        window = items[offset : offset + count]
        out = {
            **data,
            key: window,
            "_pagination": {
                "field": key,
                "offset": offset,
                "returned": len(window),
                "total": total,
                "next_offset": offset + len(window) if offset + len(window) < total else None,
            },
        }
        if _encoded_size(out) <= max_bytes or count <= 1:
            return out
        count //= 2


def _adapt(result: Any, offset: int, max_bytes: int) -> Any:
    code = getattr(result, "error_code", None)
    if code:
        raise ToolError(f"Pyrus вернул ошибку {code}: {getattr(result, 'error', '') or '—'}")
    return _paginate(_to_plain(result), offset, max_bytes)


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
    cls = getattr(pyrus_requests, spec.request)
    try:
        return cls(**payload)
    except TypeError as exc:
        keys, required = _spec_keys(spec)
        raise ToolError(
            f"{spec.method}: неверный состав request ({exc}). "
            f"Валидные ключи: {', '.join(sorted(keys)) or '—'}. "
            f"Обязательные: {', '.join(required) or 'нет'}."
        ) from exc


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
    """Одна строка таблицы → один инструмент MCP.

    Порядок в каждом замыкании обязателен: _check_offset и _check_guard идут
    до вызова Pyrus. offset проверяется в _paginate уже после вызова, и для
    пишущего инструмента это означало бы «запись состоялась, а модель получила
    только претензию к своему аргументу».
    """
    method = getattr(client, spec.method)
    description = _describe(spec, mode)

    keys, required = _spec_keys(spec)

    match spec.shape:
        case Shape.PLAIN if not keys:

            def tool(offset: int = 0) -> Any:
                _check_offset(offset)
                return _adapt(method(), offset, max_bytes)

        case Shape.PLAIN:

            def tool(options: dict | None = None, offset: int = 0) -> Any:  # type: ignore[misc]
                _check_offset(offset)
                payload = options or {}
                _check_guard(spec, payload, mode)
                return _adapt(_call_plain(client, spec, [], payload), offset, max_bytes)

        case Shape.BY_ID if not keys:

            def tool(entity_id: int, offset: int = 0) -> Any:  # type: ignore[misc]
                _check_offset(offset)
                return _adapt(method(entity_id), offset, max_bytes)

        case Shape.BY_ID:

            def tool(entity_id: int, options: dict | None = None, offset: int = 0) -> Any:  # type: ignore[misc]
                _check_offset(offset)
                payload = options or {}
                _check_guard(spec, payload, mode)
                return _adapt(
                    _call_plain(client, spec, [entity_id], payload), offset, max_bytes
                )

        case Shape.REQ:

            def tool(request: dict, offset: int = 0) -> Any:  # type: ignore[misc]
                _check_offset(offset)
                _check_guard(spec, request, mode)
                return _adapt(method(_build_request(spec, request)), offset, max_bytes)

        # Схема не врёт: если у request-класса есть обязательные поля,
        # параметр request обязателен и в сигнатуре инструмента.
        case Shape.ID_REQ if required:

            def tool(entity_id: int, request: dict, offset: int = 0) -> Any:  # type: ignore[misc]
                _check_offset(offset)
                _check_guard(spec, request, mode)
                return _adapt(method(entity_id, _build_request(spec, request)), offset, max_bytes)

        case Shape.ID_REQ:

            def tool(entity_id: int, request: dict | None = None, offset: int = 0) -> Any:  # type: ignore[misc]
                _check_offset(offset)
                payload = request or {}
                _check_guard(spec, payload, mode)
                return _adapt(method(entity_id, _build_request(spec, payload)), offset, max_bytes)

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
            result = client.download_file(file_id)
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
            return _adapt(client.upload_file(str(target)), 0, max_bytes)

        mcp.tool(upload_file)

    if "set_avatar" in specs:

        def set_avatar(member_id: int, file_guid: str, external_avatar_id: str | None = None) -> Any:
            """Назначить аватар пользователю. file_guid — идентификатор,
            полученный от upload_file, а не путь на диске."""
            return _adapt(
                client.set_avatar(member_id, file_guid, external_avatar_id), 0, max_bytes
            )

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


def _die(message: str) -> None:
    print(f"pyrus-mcp: {message}", file=sys.stderr)
    raise SystemExit(2)


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
        raise  # pragma: no cover
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
        raise  # pragma: no cover


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
