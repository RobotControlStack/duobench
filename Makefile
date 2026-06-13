PYSRC = src/duobench
MYPYSRC = src/duobench

# Python
pycheckformat:
	isort --check-only ${PYSRC}
	black --check ${PYSRC}

pyformat:
	isort ${PYSRC}
	black ${PYSRC}

pylint: ruff mypy

ruff:
	ruff check ${PYSRC}

mypy:
	mkdir -p .mypy_cache
	mypy ${MYPYSRC} --cache-dir=.mypy_cache --install-types --non-interactive --no-namespace-packages

pytest:
	pytest -vv

bump:
	cz bump

commit:
	cz commit

.PHONY: pycheckformat pyformat pylint ruff mypy pytest bump commit
