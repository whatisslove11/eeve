class StatStorage:
    def __init__(self, **kwargs):
        self._stats = kwargs

    def __add__(self, other):
        if not isinstance(other, StatStorage):
            raise TypeError(f"Cannot add StatStorage with {type(other)}")

        self_keys = set(self._stats.keys())
        other_keys = set(other._stats.keys())

        if self_keys != other_keys:
            raise ValueError(
                f"Object keys do not match. Self: {self_keys}, Other: {other_keys}"
            )

        result_stats = {}
        for key in self._stats:
            result_stats[key] = self._stats[key] + other._stats[key]

        return StatStorage(**result_stats)

    def __getattr__(self, name):
        if name in self._stats:
            return self._stats[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        if name == "_stats":
            super().__setattr__(name, value)
        else:
            self._stats[name] = value

    def __repr__(self):
        stats_str = ", ".join(f"{k}={v}" for k, v in self._stats.items())
        return f"StatStorage({stats_str})"

    def to_dict(self):
        return self._stats.copy()

    def to_json(self):
        import json

        return json.dumps(self._stats)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
