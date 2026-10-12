"""归因模型基类与注册机制。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from mta.data import AttributionResult, Journey

_REGISTRY: dict[str, type[BaseAttributionModel]] = {}


def register(cls: type[BaseAttributionModel]) -> type[BaseAttributionModel]:
    _REGISTRY[cls.name] = cls
    return cls


def get_model(name: str, **kwargs) -> BaseAttributionModel:
    if name not in _REGISTRY:
        raise KeyError(
            f"未知模型 '{name}'，可选: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


class BaseAttributionModel(ABC):
    """所有归因模型的统一接口。

    子类实现 :meth:`attribute`，输入转化路径列表，返回
    :class:`AttributionResult`（各渠道得分，无需归一化）。
    """

    name: str = "base"

    @abstractmethod
    def attribute(self, journeys: Sequence[Journey]) -> AttributionResult: ...

    # ---- 共用工具 ----
    @staticmethod
    def _channels(journeys: Sequence[Journey]) -> list[str]:
        channels: set[str] = set()
        for j in journeys:
            channels.update(j.touchpoints)
        return sorted(channels)
