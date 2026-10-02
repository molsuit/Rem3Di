from enum import Enum


class LabelScalingType(str, Enum):
    NONE = "none"
    Z = "z"
    LOG_Z = "log_z"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return None
        if isinstance(value, LabelScalingType):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"none", "identity"}:
                return cls.NONE
            if v in {"z", "standard", "standardize"}:
                return cls.Z
            if v in {"log", "log_z", "log-standardize", "log-standardise"}:
                return cls.LOG_Z
        # Returning None lets Pydantic raise its usual validation error
        return None
