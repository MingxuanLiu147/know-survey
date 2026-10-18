.PHONY: quality style

check_dirs := scripts src

quality:
	ruff check $(check_dirs) setup.py
	ruff format --check $(check_dirs) setup.py

style:
	ruff check $(check_dirs) setup.py --fix
	ruff format $(check_dirs) setup.py
