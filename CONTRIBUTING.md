# Contributing

Bug reports and pull requests are welcome.

## Before opening a pull request

```
python -m pytest
cd frontend && npm test && npm run build
```

Changes to Doppler extraction, likelihoods or the status rules also need the synthetic benchmark. Compare its results with `docs/validation_results.json`, and report anything that changes how often a location is claimed or how often it is wrong:

```
python -m backend.validation.benchmark --trials 20
```

## Code style

- No comments or docstrings in source. Explain reasoning in the pull request, and document behaviour in `docs/`.
- Type hints on all Python functions, and strict TypeScript.
- Keep the honesty rules intact. Never present a coordinate, match or probability that the evidence does not support. [docs/architecture.md](docs/architecture.md) explains why each rule exists.
