# Retrieval Evaluation — Version Comparison (pool 100, k 3)

Recall / Precision / MRR / nDCG / Accuracy / Complete-rate per ratel-ai-core version.
Δ = newest − oldest version. `—` = not run for that bucket at this pool/k.
`n` is per version when sample sizes differ (label suffix).

## BFCL

_pool 100, k 3_

### Recall

| subset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| multiple | 200 | 0.970 | 0.985 | +0.015 |
| simple | 399 | 0.980 | 0.987 | +0.008 |

### Precision

| subset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| multiple | 200 | 0.341 | 0.328 | -0.012 |
| simple | 399 | 0.345 | 0.329 | -0.016 |

### MRR

| subset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| multiple | 200 | 0.941 | 0.956 | +0.015 |
| simple | 399 | 0.951 | 0.945 | -0.006 |

### nDCG

| subset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| multiple | 200 | 0.948 | 0.963 | +0.015 |
| simple | 399 | 0.959 | 0.956 | -0.002 |

### Accuracy

| subset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| multiple | 200 | 0.970 | 0.985 | +0.015 |
| simple | 399 | 0.980 | 0.987 | +0.008 |

### Complete-rate

| subset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| multiple | 200 | 0.970 | 0.985 | +0.015 |
| simple | 399 | 0.980 | 0.987 | +0.008 |

### Run timestamps

| version | timestamp |
| --- | --- |
| `0.2.0` | 2026-06-22T21:21:50.709638+00:00 |
| `0.3.0-semantic.1` | 2026-06-25T21:25:18.728952+00:00 |

## SR-Agents

_pool 100, k 3_

### Recall

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 | 0.598 | 0.860 | +0.262 |
| champ | 0.2.0:223 / semantic.1:100 | 0.393 | 0.929 | +0.536 |
| logicbench | 0.2.0:760 / semantic.1:100 | 0.305 | 0.500 | +0.195 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 | 0.937 | 0.940 | +0.003 |
| theoremqa | 0.2.0:747 / semantic.1:100 | 0.771 | 0.990 | +0.219 |
| toolqa | 0.2.0:1430 / semantic.1:100 | 0.770 | 0.820 | +0.050 |
| all | 0.2.0:5400 / semantic.1:600 | 0.687 | 0.840 | +0.153 |

### Precision

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 | 0.534 | 0.773 | +0.240 |
| champ | 0.2.0:223 / semantic.1:100 | 0.227 | 0.537 | +0.309 |
| logicbench | 0.2.0:760 / semantic.1:100 | 0.106 | 0.167 | +0.061 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 | 0.312 | 0.313 | +0.001 |
| theoremqa | 0.2.0:747 / semantic.1:100 | 0.271 | 0.330 | +0.059 |
| toolqa | 0.2.0:1430 / semantic.1:100 | 0.334 | 0.273 | -0.061 |
| all | 0.2.0:5400 / semantic.1:600 | 0.326 | 0.399 | +0.072 |

### MRR

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 | 0.869 | 0.995 | +0.126 |
| champ | 0.2.0:223 / semantic.1:100 | 0.457 | 0.890 | +0.433 |
| logicbench | 0.2.0:760 / semantic.1:100 | 0.237 | 0.405 | +0.168 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 | 0.874 | 0.887 | +0.012 |
| theoremqa | 0.2.0:747 / semantic.1:100 | 0.694 | 0.983 | +0.289 |
| toolqa | 0.2.0:1430 / semantic.1:100 | 0.716 | 0.730 | +0.014 |
| all | 0.2.0:5400 / semantic.1:600 | 0.700 | 0.815 | +0.115 |

### nDCG

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 | 0.663 | 0.916 | +0.253 |
| champ | 0.2.0:223 / semantic.1:100 | 0.380 | 0.888 | +0.508 |
| logicbench | 0.2.0:760 / semantic.1:100 | 0.255 | 0.429 | +0.175 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 | 0.891 | 0.900 | +0.010 |
| theoremqa | 0.2.0:747 / semantic.1:100 | 0.714 | 0.985 | +0.271 |
| toolqa | 0.2.0:1430 / semantic.1:100 | 0.730 | 0.753 | +0.023 |
| all | 0.2.0:5400 / semantic.1:600 | 0.665 | 0.812 | +0.147 |

### Accuracy

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 | 0.939 | 1.000 | +0.061 |
| champ | 0.2.0:223 / semantic.1:100 | 0.547 | 0.980 | +0.433 |
| logicbench | 0.2.0:760 / semantic.1:100 | 0.305 | 0.500 | +0.195 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 | 0.937 | 0.940 | +0.003 |
| theoremqa | 0.2.0:747 / semantic.1:100 | 0.771 | 0.990 | +0.219 |
| toolqa | 0.2.0:1430 / semantic.1:100 | 0.770 | 0.820 | +0.050 |
| all | 0.2.0:5400 / semantic.1:600 | 0.765 | 0.872 | +0.106 |

### Complete-rate

| dataset | n | 0.2.0 | 0.3.0-semantic.1 | Δ (0.3.0-semantic.1−0.2.0) |
| --- | --- | --- | --- | --- |
| bigcodebench | 0.2.0:1140 / semantic.1:100 | 0.232 | 0.620 | +0.388 |
| champ | 0.2.0:223 / semantic.1:100 | 0.269 | 0.860 | +0.591 |
| logicbench | 0.2.0:760 / semantic.1:100 | 0.305 | 0.500 | +0.195 |
| medcalcbench | 0.2.0:1100 / semantic.1:100 | 0.937 | 0.940 | +0.003 |
| theoremqa | 0.2.0:747 / semantic.1:100 | 0.771 | 0.990 | +0.219 |
| toolqa | 0.2.0:1430 / semantic.1:100 | 0.770 | 0.820 | +0.050 |
| all | 0.2.0:5400 / semantic.1:600 | 0.604 | 0.788 | +0.184 |

### Run timestamps

| version | timestamp |
| --- | --- |
| `0.2.0` | 2026-06-24T11:37:32.211566+00:00 |
| `0.3.0-semantic.1` | 2026-06-26T12:04:54.263179+00:00 |

