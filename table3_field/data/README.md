# Field test data (not included)

Table 3 is computed on 10 patches of the Bouguer gravity anomaly of the Mangampeta region (southern Cuddapah
basin, Andhra Pradesh, India). The data were provided by the Geological Survey of India (Ganguli et al., 2019)
and cannot be redistributed, so they are not part of this repository.
<!-- TODO: state how readers can request the data (Geological Survey of India / corresponding author) -->

With the data, put the 10 files in `data/test/` and check them:

```bash
python scripts/check_data.py
```

| File | Content | MD5 |
|---|---|---|
| `data14.mat` | variable `d`, 32 x 32 nodes, 400 m spacing | `63151da22fbb3f98b02fab58e5fc2933` |
| `data16.mat` | 〃 | `0a5947cfbf9c35dff41107a61a6c1099` |
| `data18.mat` | 〃 | `52a7baad81ff0aa151b2e0b435bad4f0` |
| `data20.mat` | 〃 | `995bb1c4b8e2840ddcbd59487d5ac2ce` |
| `data22.mat` | 〃 | `d69eafc6ae3379e7c99fb16536d31d73` |
| `data24.mat` | 〃 | `627f7b19adcdedcaed9256dcda69221d` |
| `data26.mat` | 〃 | `eb2f9d95a8c72984a817138a1acfd82c` |
| `data28.mat` | 〃 | `b1318531e2e7ac557185a6d6a8479c8a` |
| `data30.mat` | 〃 | `c9a2fa3e7b13f009d33d4c70566e220a` |
| `data32.mat` | 〃 | `83ee18dc7e18428c1504ed89eb53882f` |

Without the data, the notebooks of this folder cannot be run, but `results/` contains the per-patch metrics
of every row of Table 3, and `notebooks/00_Reproduce_Table3.ipynb` assembles and checks the table from them.
