class FakeResponse:
    """Заменяет объекты pyrus.models.responses: тот же контракт по
    error_code / original_response, что читает адаптер."""

    def __init__(self, payload=None, error_code=None, error=None):
        self.original_response = payload or {}
        self.error_code = error_code
        self.error = error


class FakeClient:
    """Записывает вызовы вместо похода в сеть."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result if result is not None else FakeResponse({"ok": True})

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.result

        return call
