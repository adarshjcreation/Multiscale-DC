# Field test data

Table 3 is computed on 10 patches of the Bouguer gravity anomaly of the Mangampeta region (southern Cuddapah
basin, Andhra Pradesh, India). The data were provided by the Geological Survey of India:

> Ganguli, S.S., Singh, S., Das, N., Maurya, D., Pal, S.K., Rama Rao, J.V., 2019. Gravity and magnetic survey in
> southwestern part of Cuddapah Basin, India and its implication for shallow crustal architecture and
> mineralization. Journal of the Geological Society of India 93, 419–430.

The 10 patches are archived on Zenodo (`field_test.zip`, DOI
[10.5281/zenodo.24340552](https://doi.org/10.5281/zenodo.24340552)), not in this git repository. From the top
folder of the repository, download them into `data/test/` and check them:

```bash
python scripts/download_data.py field_test     # downloads, checks the MD5 and copies the files to data/test/
python table3_field/scripts/check_data.py      # checks every file against the MD5 below
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

`d` is a vector of 1,024 values (float64), read as `d.reshape(32, 32)` like the synthetic grids. The files hold
no coordinates. If you use these data, please cite Ganguli et al. (2019) and the Zenodo record.

Without the data, the notebooks of this folder cannot be run, but `results/` contains the per-patch metrics
of every row of Table 3, and `notebooks/00_Reproduce_Table3.ipynb` assembles and checks the table from them.
