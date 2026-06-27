# Retrieval Evaluation — Version Comparison (pool 100, k 3)

Recall / Precision / MRR / nDCG / Accuracy / Complete-rate per ratel-ai-core version.
Δ = newest − oldest version. `—` = not run for that bucket at this pool/k.
`n` is per version when sample sizes differ (label suffix).

## BFCL

_pool 100, k 3_

### Recall

| subset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| multiple | 200 | 0.970 | 0.985 | 0.735 | -0.235 |
| simple | 399 | 0.980 | 0.987 | 0.697 | -0.283 |

### Precision

| subset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| multiple | 200 | 0.341 | 0.328 | 0.245 | -0.096 |
| simple | 399 | 0.345 | 0.329 | 0.232 | -0.113 |

### MRR

| subset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| multiple | 200 | 0.941 | 0.956 | 0.628 | -0.313 |
| simple | 399 | 0.951 | 0.945 | 0.619 | -0.332 |

### nDCG

| subset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| multiple | 200 | 0.948 | 0.963 | 0.655 | -0.293 |
| simple | 399 | 0.959 | 0.956 | 0.639 | -0.320 |

### Accuracy

| subset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| multiple | 200 | 0.970 | 0.985 | 0.735 | -0.235 |
| simple | 399 | 0.980 | 0.987 | 0.697 | -0.283 |

### Complete-rate

| subset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| multiple | 200 | 0.970 | 0.985 | 0.735 | -0.235 |
| simple | 399 | 0.980 | 0.987 | 0.697 | -0.283 |

### Run timestamps

| version | timestamp |
| --- | --- |
| `0.2.0` | 2026-06-22T21:21:50.709638+00:00 |
| `0.3.0-semantic.1` | 2026-06-25T21:25:18.728952+00:00 |
| `0.3.0-semantic.2` | 2026-06-26T15:03:04.412594+00:00 |

## SR-Agents

_pool 100, k 3_

### Recall

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 / semantic.2:100 | 0.598 | 0.860 | 0.753 | +0.155 |
| champ | 0.2.0:223 / semantic.1:100 / semantic.2:100 | 0.393 | 0.929 | 0.920 | +0.527 |
| logicbench | 0.2.0:760 / semantic.1:100 / semantic.2:100 | 0.305 | 0.500 | 0.670 | +0.365 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 / semantic.2:100 | 0.937 | 0.940 | 0.790 | -0.147 |
| theoremqa | 0.2.0:747 / semantic.1:100 / semantic.2:100 | 0.771 | 0.990 | 0.980 | +0.209 |
| toolqa | 0.2.0:1430 / semantic.1:100 / semantic.2:100 | 0.770 | 0.820 | 0.420 | -0.350 |
| all | 0.2.0:5400 / semantic.1:600 / semantic.2:600 | 0.687 | 0.840 | 0.756 | +0.068 |

### Precision

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 / semantic.2:100 | 0.534 | 0.773 | 0.687 | +0.153 |
| champ | 0.2.0:223 / semantic.1:100 / semantic.2:100 | 0.227 | 0.537 | 0.527 | +0.299 |
| logicbench | 0.2.0:760 / semantic.1:100 / semantic.2:100 | 0.106 | 0.167 | 0.223 | +0.118 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 / semantic.2:100 | 0.312 | 0.313 | 0.263 | -0.049 |
| theoremqa | 0.2.0:747 / semantic.1:100 / semantic.2:100 | 0.271 | 0.330 | 0.327 | +0.056 |
| toolqa | 0.2.0:1430 / semantic.1:100 / semantic.2:100 | 0.334 | 0.273 | 0.140 | -0.194 |
| all | 0.2.0:5400 / semantic.1:600 / semantic.2:600 | 0.326 | 0.399 | 0.361 | +0.035 |

### MRR

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 / semantic.2:100 | 0.869 | 0.995 | 0.947 | +0.078 |
| champ | 0.2.0:223 / semantic.1:100 / semantic.2:100 | 0.457 | 0.890 | 0.910 | +0.453 |
| logicbench | 0.2.0:760 / semantic.1:100 / semantic.2:100 | 0.237 | 0.405 | 0.537 | +0.300 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 / semantic.2:100 | 0.874 | 0.887 | 0.690 | -0.184 |
| theoremqa | 0.2.0:747 / semantic.1:100 / semantic.2:100 | 0.694 | 0.983 | 0.968 | +0.274 |
| toolqa | 0.2.0:1430 / semantic.1:100 / semantic.2:100 | 0.716 | 0.730 | 0.370 | -0.346 |
| all | 0.2.0:5400 / semantic.1:600 / semantic.2:600 | 0.700 | 0.815 | 0.737 | +0.037 |

### nDCG

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 / semantic.2:100 | 0.663 | 0.916 | 0.816 | +0.153 |
| champ | 0.2.0:223 / semantic.1:100 / semantic.2:100 | 0.380 | 0.888 | 0.886 | +0.506 |
| logicbench | 0.2.0:760 / semantic.1:100 / semantic.2:100 | 0.255 | 0.429 | 0.571 | +0.316 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 / semantic.2:100 | 0.891 | 0.900 | 0.716 | -0.175 |
| theoremqa | 0.2.0:747 / semantic.1:100 / semantic.2:100 | 0.714 | 0.985 | 0.971 | +0.258 |
| toolqa | 0.2.0:1430 / semantic.1:100 / semantic.2:100 | 0.730 | 0.753 | 0.383 | -0.347 |
| all | 0.2.0:5400 / semantic.1:600 / semantic.2:600 | 0.665 | 0.812 | 0.724 | +0.059 |

### Accuracy

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 / semantic.2:100 | 0.939 | 1.000 | 0.980 | +0.041 |
| champ | 0.2.0:223 / semantic.1:100 / semantic.2:100 | 0.547 | 0.980 | 0.980 | +0.433 |
| logicbench | 0.2.0:760 / semantic.1:100 / semantic.2:100 | 0.305 | 0.500 | 0.670 | +0.365 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 / semantic.2:100 | 0.937 | 0.940 | 0.790 | -0.147 |
| theoremqa | 0.2.0:747 / semantic.1:100 / semantic.2:100 | 0.771 | 0.990 | 0.980 | +0.209 |
| toolqa | 0.2.0:1430 / semantic.1:100 / semantic.2:100 | 0.770 | 0.820 | 0.420 | -0.350 |
| all | 0.2.0:5400 / semantic.1:600 / semantic.2:600 | 0.765 | 0.872 | 0.803 | +0.038 |

### Complete-rate

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | 0.3.0-semantic.2 | Δ (0.3.0-semantic.2−0.2.0) |
| --- | --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 / semantic.2:100 | 0.232 | 0.620 | 0.440 | +0.208 |
| champ | 0.2.0:223 / semantic.1:100 / semantic.2:100 | 0.269 | 0.860 | 0.830 | +0.561 |
| logicbench | 0.2.0:760 / semantic.1:100 / semantic.2:100 | 0.305 | 0.500 | 0.670 | +0.365 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 / semantic.2:100 | 0.937 | 0.940 | 0.790 | -0.147 |
| theoremqa | 0.2.0:747 / semantic.1:100 / semantic.2:100 | 0.771 | 0.990 | 0.980 | +0.209 |
| toolqa | 0.2.0:1430 / semantic.1:100 / semantic.2:100 | 0.770 | 0.820 | 0.420 | -0.350 |
| all | 0.2.0:5400 / semantic.1:600 / semantic.2:600 | 0.604 | 0.788 | 0.688 | +0.084 |

### Run timestamps

| version | timestamp |
| --- | --- |
| `0.2.0` | 2026-06-24T11:37:32.211566+00:00 |
| `0.3.0-semantic.1` | 2026-06-26T12:04:54.263179+00:00 |
| `0.3.0-semantic.2` | 2026-06-27T12:34:12.120725+00:00 |

