.PHONY: docs cleandocs

docs:
	cd docs && make html SPHINXOPTS="-d _build/doctrees"

cleandocs:
	cd docs && make clean
