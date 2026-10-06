PYSRC = src/duobench
MYPYSRC = src/duobench
WHEELHOUSE = dist
PYPI_REPOSITORY ?= pypi

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

buildwheel:
	rm -rf ${WHEELHOUSE}
	uv build --wheel --out-dir ${WHEELHOUSE} .
	twine check ${WHEELHOUSE}/*.whl

uploadwheel:
	twine upload --repository ${PYPI_REPOSITORY} ${WHEELHOUSE}/*.whl

.PHONY: pycheckformat pyformat pylint ruff mypy pytest bump commit buildwheel uploadwheel
