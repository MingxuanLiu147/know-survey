# Third-party notices

Know-Surgery builds on OpenUnlearning. The root [LICENSE](LICENSE) retains its
MIT copyright notice. The following upstream sources inform code ports and
method adaptations; their available license texts are retained in `LICENSES/`.

| Source | Reference revision | Attribution |
|---|---|---|
| [OpenUnlearning](https://github.com/locuslab/open-unlearning/tree/4ad738aaf60f6a4385f6e2506d01da99e76c31f3) | `4ad738aaf60f6a4385f6e2506d01da99e76c31f3` | Inherited framework; root MIT license |
| [EasyEdit](https://github.com/zjunlp/EasyEdit/tree/14cea8245f06715684592ab55184939b99d70784) | `14cea8245f06715684592ab55184939b99d70784` | WISE-Merge mechanism adaptation, SERAC causal handling and algorithm/README references; [upstream MIT text](LICENSES/EasyEdit.txt) |
| [SERAC](https://github.com/eric-mitchell/serac/tree/c963a39fa77a743f054bed372056fa1a66e224a4) | `c963a39fa77a743f054bed372056fa1a66e224a4` | Supervised scope-encoder/counterfactual-model mechanism adapted for causal LMs; Copyright (c) 2022 Eric Mitchell, [MIT text](LICENSES/SERAC.txt) |
| [ROME](https://github.com/kmeng01/rome/tree/0874014cd9837e4365f3e6f3c71400ef11509e04) | `0874014cd9837e4365f3e6f3c71400ef11509e04` | `src/trainer/edit/utils/nethook.py` port and method reference; [MIT text](LICENSES/ROME.txt) |
| [MEMIT](https://github.com/kmeng01/memit/tree/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b) | `80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b` | Source for rebuilt value-optimization and layer-statistics adapters; [MIT text](LICENSES/MEMIT.txt) |
| [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit/tree/b84624f44dfe8fc6cd9e41df916c44124a0c46dc) | `b84624f44dfe8fc6cd9e41df916c44124a0c46dc` | Null-space projection, history-Gram update, target objective and context protocol reference; Copyright (c) 2025 jianghc, [MIT text](LICENSES/AlphaEdit.txt) |
| [IKE](https://github.com/pkunlp-icler/IKE/tree/da58c842cd95628f281f474bc432a81cbd1cfd1e) | `da58c842cd95628f281f474bc432a81cbd1cfd1e` | Demonstration protocol reference; implemented locally without vendoring upstream Python. No upstream license text is redistributed. |
| [AnyEdit](https://github.com/jianghoucheng/AnyEdit/tree/057a77f185f7ffb55818f6bd9add37f43bb447e7) | `057a77f185f7ffb55818f6bd9add37f43bb447e7` | Autoregressive knowledge editing reference; [MIT text](LICENSES/AnyEdit.txt) |

Method names identify algorithms, not universal upstream checkpoint or benchmark
compatibility. See [editing](docs/editing.md) for supported settings and required
auxiliary assets. The IKE demonstration protocol is independently implemented
in the existing editor; no MIT license is assigned to the upstream IKE repository.

The distributable package excludes the UnKE optimization implementation.
AnyEdit uses MEMIT or AlphaEdit backends. The separate UnKE-format dataset adapter
supplies no benchmark data or additional dataset rights.

Datasets, base model weights and auxiliary checkpoints have their own licenses
and are separate inputs; this source package does not redistribute them.

The WebUI bundles Inter font files (weights 400, 500 and 600) under the
[SIL Open Font License 1.1](webui/frontend/public/fonts/inter/Inter-LICENSE.txt).
The font copyright and source information accompany those files.
