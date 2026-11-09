# Website

The project website, https://colalab.ai/pinnseek/, is built from this folder with
MkDocs Material and published by `.github/workflows/docs.yml` on every push to `main` that touches
`docs/`.

```bash
uv venv docs/.venv-docs && uv pip install --python docs/.venv-docs -r docs/requirements.txt
docs/.venv-docs/bin/mkdocs serve -f docs/mkdocs.yml        # preview at http://127.0.0.1:8820
docs/.venv-docs/bin/mkdocs build --strict -f docs/mkdocs.yml
```

| Path | Content |
|---|---|
| `mkdocs.yml` | site settings and navigation; `extra.apply_url` is the trial application form |
| `overrides/home.html`, `content/assets/home/` | the landing page: its sections, styles (`home.css`), the cards of its numbers (`home.js`) and the paper's figures |
| `overrides/partials/` | data blocks of the landing page (the comparison chart and the benchmark gallery) and `header.html`, Material's header with the site name linked home |
| `content/problems.md`, `problems/` | the 18 benchmark problems |
| `content/assets/` | `theme.css` (site colours), `pf.css` and `pf.js` (results pages), figures, animations |
| `content/assets/fields/` | the predicted and error fields after each round, shown when hovering a round of a discovery chart |
| `content/files/` | the discovered PINN solvers, the drivers of the four comparison methods and the code of every candidate on the discovery charts (`candidates/`) |
| `overrides/pf.html` | template of the results pages |

The results pages and their assets are generated from the paper's data and committed as they are;
edit `theme.css` by hand.
