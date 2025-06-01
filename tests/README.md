To set up and run tests:

```bash
# Activate virtual environment
source venv/bin/activate

# Install test dependencies (if not already installed)
pip install pytest pytest-qt pytest-mock

# Create test data
mkdir -P "test_data/album1"
python tests/create_test_data.py

# Run tests
pytest -rP -v tests
```
