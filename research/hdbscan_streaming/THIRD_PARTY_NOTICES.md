# Third-Party Notices

Project-owned source code is released under the repository MIT License:
[`../../LICENSE`](../../LICENSE). This notice does not relicense any
third-party source or dataset-derived file.

The research package vendors source code for two optional comparison backends.
Their original license files are retained beside the corresponding source.

## Flexible Clustering / FISHDBC

- Location: `vendor/flexible_clustering/`
- Upstream paper: <https://arxiv.org/abs/1910.07283>
- Copyright: Symantec Corporation, 2017-2018
- License: BSD 3-Clause
- Notice: `vendor/flexible_clustering/LICENSE`

The package contains a Cython extension. `scripts/setup_research_env.sh`
builds it inside the research virtual environment.

## Fast HDBSCAN

- Location: `vendor/fast_hdbscan/`
- Upstream project: <https://github.com/TutteInstitute/fast_hdbscan>
- Copyright: Tutte Institute for Mathematics and Computing, 2023
- License: BSD 2-Clause
- Notice: `vendor/fast_hdbscan/LICENSE`

## FSDD

The raw Free Spoken Digit Dataset is not redistributed here. The committed
embedding fixtures and manifests are derived artifacts. See
`data/fsdd/PROVENANCE.md` for the upstream URL, checksum, and CC BY-SA 4.0
notice.
