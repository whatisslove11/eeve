import torch.nn as nn


def is_module_exists(model: nn.Module, pattern: str) -> nn.Module | None:
    return next(
        (
            model.get_submodule(n.rpartition(".")[0])
            for n, _ in model.named_parameters()
            if pattern in n
        ),
        None,
    )


def get_first_layer_by_type(
    model: nn.Module, module_type: type, return_shape: bool = True
) -> tuple:
    for name, param in model.named_parameters():
        parent_path = name.rpartition(".")[0]
        module = model.get_submodule(parent_path)

        if isinstance(module, module_type):
            if return_shape:
                return parent_path, param.shape
            return parent_path, None

    return None, None


def find_module_by_pattern(
    model: nn.Module,
    name_patterns: list[str],
    module_type: type,
) -> str | None:
    for name, _ in model.named_parameters():
        if not any(pattern in name for pattern in name_patterns):
            continue
        parent_path = name.rpartition(".")[0]
        if isinstance(model.get_submodule(parent_path), module_type):
            return parent_path
    return None


def find_layers_by_shape_and_type(
    model: nn.Module, module_type: type, target_shape: tuple
) -> list[str]:
    found_layers = []

    for name, param in model.named_parameters():
        if param.shape != target_shape:
            continue

        parent_path = name.rpartition(".")[0]
        if parent_path in found_layers:
            continue

        if isinstance(model.get_submodule(parent_path), module_type):
            found_layers.append(parent_path)
    return found_layers
