# Upstream Provenance

- **Source**: https://github.com/EvolvingLMMs-Lab/lmms-eval
- **Commit**: `6619ef618960f523489b5d7ffa5e9611b4715215` (2026-07-21)
- **Sub-path**: —
- **Cloned**: 2026-07-24
- **Release layout**: standalone experiment scripts; this release contains no `relprobe` package or packaging metadata.
- **Local divergence starts**: `601c731c22707c89ebdc5cad0e39c4367101b7d9`

The commit and clone details above are retained provenance records; the original source history is not bundled here, so the submission commits do not verify those upstream records. The scripts use synthetic data when `--input` is omitted and mark those outputs with `demo: true`. These runs do not evaluate actual models or datasets. The `semantic_entropy` output key is retained for compatibility, but computes entropy over normalized answer strings rather than semantic clusters. Efficiency results measure scoring only and exclude model inference.
