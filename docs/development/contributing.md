# Contributing

## Setup

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/a2c_ase.git
cd a2c_ase

# Install with dev dependencies
pip install -e ".[dev,test]"

# Set up pre-commit hooks
pre-commit install

# Create branch
git checkout -b feature/your-feature-name
```

---

## Development

### Code Quality

```bash
ruff check        # Lint
ruff format       # Format
ty check          # Type check
pytest            # Test
```

### Documentation

```bash
pip install -e ".[docs]"
mkdocs serve      # Preview
```

---

## Code Standards

1. **PEP 8**: Enforced by `ruff`
2. **Type hints**: Required for all functions
3. **Docstrings**: NumPy-style format
4. **Tests**: Required for new features

### Docstring Template

```python
def function(param: int) -> bool:
    """Brief description.

    Parameters
    ----------
    param : int
        Description

    Returns
    -------
    bool
        Description
    """
```

---

## Testing

### Write Tests

```python
def test_feature():
    """Test description."""
    result = function(input)
    assert result == expected
```

### Run Tests

```bash
pytest                    # All tests
pytest -k "pattern"       # Match pattern
pytest --cov=a2c_ase      # With coverage
```

---

## Submit Changes

### Commit

```bash
git add .
git commit -m "feat: description"
```

**Prefixes**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

### Push & PR

```bash
git push origin feature/your-feature-name
```

Create PR on GitHub with clear description.

### PR Checklist

- [ ] Tests pass
- [ ] Coverage >= 60%
- [ ] Type hints
- [ ] Docstrings

---

## Reporting Issues

**Bug reports**: Include description, steps to reproduce, environment, minimal example

**Feature requests**: Include description, use case, motivation

---

## Resources

- [Code](https://github.com/abhijeetgangan/a2c_ase/tree/main/a2c_ase)
- [Tests](https://github.com/abhijeetgangan/a2c_ase/tree/main/tests)
- [User Guide](../user-guide/workflow.md)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

**Thank you!**
