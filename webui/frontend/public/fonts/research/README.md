# Workbench fonts

English text uses IBM Plex Sans 3.201 (variable 100–700), and commands use IBM Plex
Mono 2.3 (400/500). Chinese text uses a Noto Sans SC 2.004-H2 subset (400/500/600),
renamed **Know Surgery SC** to identify the derivative.

The Chinese subset preserves the original outlines and weights. It contains the
GB2312 repertoire, interface characters, punctuation, and selected mathematical
symbols. Other characters use the platform fallbacks in `fonts.css`.

The six WOFF2 files total 3.10 MiB. `fonts.css` defines `--font-ui` and `--font-code`
with `font-display: swap`. All assets are served locally; no external font service
is required.

Fonts are distributed under SIL Open Font License 1.1. Keep [IBM's license](IBM-Plex-OFL.txt)
and [Noto's license](Noto-Sans-SC-OFL.txt) with the assets. [SOURCES.json](SOURCES.json)
records source URLs, SHA256 hashes, and modifications. Upstream projects:
[IBM Plex](https://github.com/IBM/plex), [Noto CJK](https://github.com/notofonts/noto-cjk).
