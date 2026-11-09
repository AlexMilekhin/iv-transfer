# Zero-Shot Transfer of Implied-Volatility Forecasts Across Option Markets

Code for the paper of the same name (Alexander Milekhin, 2026). It goes from the exchanges' raw files to every table, figure and
number in the paper.

```
markets/     one script per market (US, Brazil, India, Hong Kong, Canada): download, universe, option chains
src/ivtransfer/  surface fits, model-ready samples, training, scoring, tables, Section 8
notebooks/   the whole pipeline on Google Colab
docs/        paper_map.md (which file and function gives each result), figure1.png
tests/       unit tests
```

## Results

A network trained on US equity options forecasts the two-week change in implied volatility in Brazil, India, Hong Kong and
Canada better than the same network trained on local data alone:

- The best US-informed model improves on the local model in every market, by 1.1 to 3.7 points of skill over the test years
  2023-25 and by 0.5 to 6.7 points in the 2026 holdout.
- A model chosen on validation data before each test year beats the local model in all four markets over 2023-25, with
  block-bootstrap intervals that exclude zero, and in three of the four in 2026. The gains are stable across training seeds.
- Over 2023-25 the networks also beat simple ridge, term-slope and gradient-boosting benchmarks; in 2026 simple rules beat the
  networks in three of the four markets.
- In a US trading test, the forecasts are informative about implied volatility but do not make delta-hedged straddles
  profitable after costs.

## Data

Public, not redistributed; each market script downloads its own data.

| Market | Source | Years |
|---|---|---|
| United States | DoltHub `post-no-preference/options` (capped at 12 Oct 2026) | 2020-26 |
| Brazil | B3 COTAHIST | 2015-26 |
| India | NSE F&O bhavcopy | 2018-26 |
| Hong Kong | HKEX DTOP stock options | 2018-26 |
| Canada | Montreal Exchange | 2017-26 |

## Running

```bash
pip install -e ".[markets]"        # Python >= 3.11

python markets/us.py --root data   # likewise br.py, india.py, hongkong.py, canada.py

python -m ivtransfer universe --root data
python -m ivtransfer surfaces --root data   # CPU, hours
python -m ivtransfer states   --root data
python -m ivtransfer train    --root data   # GPU, hours
python -m ivtransfer initial  --root data   # GPU
python -m ivtransfer paper    --root data   # CPU, ~2 h
python -m ivtransfer trading  --root data   # CPU, ~1 h
python -m ivtransfer figures  --root data
```

Each stage is resumable. Results go to `data/results/paper/`; `paper_numbers.csv` lists every number quoted in the text.
Seeds are fixed throughout, and scoring checks every reloaded model against its saved forecasts before reporting.

## Citation

```bibtex
@article{milekhin2026transfer,
  title  = {Zero-Shot Transfer of Implied-Volatility Forecasts Across Option Markets},
  author = {Milekhin, Alexander},
  year   = {2026},
  note   = {Preprint}
}
```

Code under the MIT licence; the data belong to their providers.
