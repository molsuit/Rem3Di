from dataclasses import asdict, fields

import yaml


def dataclass_from_dict(data: dict, cls):
    """Convert a dictionary to a dataclass instance."""

    fieldtypes = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for field, field_type in fieldtypes.items():
        value = data.get(field)
        if isinstance(value, dict) and hasattr(field_type, "__dataclass_fields__"):
            # If the value is a dictionary and the field type is a dataclass, recursively call from_dict
            kwargs[field] = dataclass_from_dict(value, field_type)
        else:
            kwargs[field] = value
    return cls(**kwargs)


def from_yaml(yaml_file: str, cls):
    with open(yaml_file) as file:
        data = yaml.safe_load(file)
    return dataclass_from_dict(data, cls)


def to_yaml(yaml_file, data):
    data_in_dict = asdict(data)
    with open(yaml_file, "w") as file:
        yaml.dump(data_in_dict, file)
