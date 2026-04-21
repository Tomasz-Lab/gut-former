# 🤝 Contributing

## 🛠️ Development

Before committing your changes:

```bash
make check    # Run all: fix + typecheck
```

### Tools

This project uses the following development tools:

| Tool | Purpose | Command |
|------|---------|---------|
| **Ruff** | Linting and formatting | `make fix` |
| **Mypy** | Static type checking | `make typecheck` |


## 📓 Jupyter Notebooks

Register the kernel to access project dependencies in notebooks:

```bash
make kernel   # Register kernel (one time)
make jupyter  # Launch Jupyter
```

Select kernel "Python (gut-former)" in your notebook.

