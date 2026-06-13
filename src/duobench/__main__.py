import pkgutil
from importlib import import_module

import duobench.tasks


def _register_duobench_tasks() -> None:
    for module in pkgutil.iter_modules(duobench.tasks.__path__):
        import_module(f"{duobench.tasks.__name__}.{module.name}")


def main() -> None:
    # Import the base RCS CLI only after DuoBench tasks are registered so the
    # existing commands can resolve `duobench/*` env ids.
    _register_duobench_tasks()
    import_module("rcs.__main__").app()


if __name__ == "__main__":
    main()
