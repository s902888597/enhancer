# Data Tracking

This repository tracks MIL code, plotting scripts, lightweight summaries, and file manifests.

Raw shared data matrices are not tracked in Git because `shared_data/` is 9.4 GB and contains files larger than GitHub's 100 MB per-file limit. The local HPC copy is indexed by:

- `docs/shared_data_manifest.tsv`: all files under `shared_data/`
- `docs/shared_data_large_files_over_95mb.tsv`: `shared_data` files that are too large for normal GitHub storage
- `docs/mil_large_files_over_95mb.tsv`: large MIL-side outputs excluded from normal GitHub storage

Each manifest row is:

```text
size_bytes    modified_time    relative_path
```
