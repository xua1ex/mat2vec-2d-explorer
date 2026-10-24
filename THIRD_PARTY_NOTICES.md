# Third-party notices

## mat2vec

`explorer/formula.py` is adapted from `mat2vec/processing/process.py` (https://github.com/materialsintelligence/mat2vec, commit 7ca8961).
The build step reads the pretrained Mat2vec embeddings from Tshitoyan et al., *Nature* 571, 95–98 (2019), doi:10.1038/s41586-019-1335-8.
The pretrained embeddings are distributed under the MIT License below. Redistribution of the pretrained weights under
the MIT License was confirmed by first author Vahe Tshitoyan via email in October 2026. This repository ships the four derived files in `artifacts/`
(including `vectors.npz`: input vectors for the 50,482 most frequent tokens and the candidates' input and output vectors,
in float16) under that licence. The original ~880 MB model files are not included.

```
MIT License

Copyright (c) 2019 Authors of Tshitoyan et al. Nature (2019)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Derived results (`public_artifacts/`)

The results-only bundle for the hosted version contains precomputed, limited-scope outputs derived from the
pretrained Mat2vec embeddings: similarity scores for 15 predefined targets, PCA coordinates and top-10 context words.
It does not contain the embedding vectors or model files and serves as a fallback when `artifacts/` is unavailable.
It is distributed under the same MIT License. See `public_artifacts/README.md`.

## MC2D (Materials Cloud two-dimensional crystals database)

Band gaps and formulas are retrieved at build time from the MC2D OPTIMADE API (https://optimade.materialscloud.org/main/mc2d),
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). MC2D does not endorse this project.

**Changes made to the MC2D data** (in `explorer/build_data.py`; the raw API response is cached unmodified in `data/mc2d_raw.json`, which is not redistributed):
- filtered to entries with a computed band gap > 0 (`_mcloud_band_gap > 0`); only `chemical_formula_reduced` and `_mcloud_band_gap` are used;
- formulas normalized to Mat2vec's convention (elements sorted alphabetically, integer stoichiometry, e.g. WS2 -> S2W);
- entries sharing a normalized formula (polymorphs) merged into one row, keeping the band-gap minimum and maximum and the entry count;
- only formulas present in the Mat2vec vocabulary are kept (591 of 1,362);
- chemical-family labels and Mat2vec scores are added; they are not MC2D data.
Mounet, N. et al. *Nat. Nanotechnol.* 13, 246–252 (2018), doi:10.1038/s41565-017-0035-5.
Campi, D., Mounet, N., Gibertini, M., Pizzi, G., Marzari, N. *ACS Nano* 17, 11268–11278 (2023).
