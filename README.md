# evidence-entropy

Information-theoretic analysis of meta-analytic trust. The engine computes
Shannon entropy, Kullback-Leibler divergence, mutual information, and
conditional entropy over the joint distribution of GRADE certainty grades and
statistical significance, stratified by Cochrane review group.

## What it does

`entropy_engine.py` provides:

- `shannon_entropy(distribution)` — H(X) in bits (input normalised; raises on
  empty / all-zero input).
- `kl_divergence(p, q)` — D_KL(P || Q) in bits, with an epsilon stabiliser for
  zero denominators.
- `mutual_information(contingency)` — I(X; Y) from a 2-D contingency table.
- `conditional_entropy(contingency)` — H(Y | X).
- `normalized_mi(mi, h_trust, h_sig)` — NMI = MI / sqrt(H_trust · H_sig).
- A pipeline (`run_pipeline`) that joins per-meta-analysis GRADE scores,
  significance verdicts, and review-group labels, then reports overall and
  per-domain entropy / MI / NMI.

The pipeline reads three external CSVs (`scores.csv`, `verdicts.csv`,
`review_groups.csv`) whose locations are resolved at import time. When those
corpus files are absent, the core information-theoretic functions still work on
in-memory inputs.

## Files

- `entropy_engine.py` — core engine (pure functions + data loaders + pipeline).
- `build_dashboard.py`, `build_e156.py` — artifact generators.
- `dashboard.html` / `index.html` — offline static dashboard.
- `tests/test_entropy.py` — unit tests for the engine; the five
  pipeline-integration tests skip when the external corpus is not present.

## Tests

```
python -m pytest -q
```

20 unit tests cover the information-theoretic functions (uniform/delta entropy,
KL known values and asymmetry, MI independence/perfect-dependence, the
I(X;Y) = H(Y) − H(Y|X) identity, and edge cases). Five integration tests
exercise the full pipeline against the external corpus and skip when it is
unavailable.

## License

MIT — see `LICENSE`.
