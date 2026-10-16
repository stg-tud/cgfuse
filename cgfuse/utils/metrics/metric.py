from typing import Any

class Metric:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError()
