#find . -iname "*.py" | xargs pylint
pytest tests.py --cov=$(pwd)