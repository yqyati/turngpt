# References and attribution

The project adapts the task and parts of the later simplified TurnGPT implementation by Erik Ekstedt and Gabriel Skantze.

- Paper: Erik Ekstedt and Gabriel Skantze (2020), *TurnGPT: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog*. Included as `TurnGPT_2020.pdf`.
- Reference code: https://github.com/ErikEkstedt/TurnGPT . Original working-project checkout: `64f7d8c3b00e2a8332bca9267e6252462cbd0407`. Its MIT notice is preserved verbatim in `TurnGPT_LICENSE.txt`.
- Reference dataset repository: https://github.com/ErikEkstedt/datasets_turntaking . Original working-project checkout: `8236f3188798381a4815250a0337a7a090137540`. The package retains the Taskmaster 2/3 and MetaLWOZ split indices used by that checkout.

The copied sources cover DailyDialog, ParlAI PersonaChat `both_original`, MultiWOZ 2.1, CCPE, selected Taskmaster 1/2/3 configurations, and MetaLWOZ. Their identities, selected configurations, and split policies are described in the README. Original source manifests and available raw hashes are retained in `data/provenance/`; `data/raw/taskmaster/manifest.json` records the downloaded Taskmaster source snapshot.

CCPE's source is https://github.com/google-research-datasets/ccpe at recorded commit `2c9cd30f33f3a154b5a27d015333679262ff36f5` (CC BY 4.0). Taskmaster's source is https://github.com/google-research-datasets/Taskmaster . The PersonaChat files follow the official ParlAI release; the MultiWOZ archive is the official 2.1 release.

Model weights are not distributed in this package. Original data and model providers retain their respective licenses and access conditions. The TurnGPT MIT notice applies to the reference code and does not relicense all included datasets.
