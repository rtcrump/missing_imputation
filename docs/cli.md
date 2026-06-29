# CLI Reference

The `missing-impute` command is installed automatically with the package.

## Commands

### `missing-impute impute`

Impute missing values in a CSV or XLSX file.

```bash
missing-impute impute INPUT -o OUTPUT [--method METHOD] [--columns COLS]
```

| Argument | Description |
|---|---|
| `INPUT` | Path to the input CSV or XLSX file |
| `-o`, `--output` | **Required.** Output CSV path |
| `-m`, `--method` | Imputation method: `knn` (default), `mean`, `median`, `mice`, `softimpute` |
| `--columns` | Comma-separated column names to impute. If omitted, defaults to FACT-E items found in the file (the included demo instrument) |

**Examples:**

```bash
# Impute your own columns with KNN (default method)
missing-impute impute patient_data.csv -o imputed.csv --columns q1,q2,q3,q4,q5

# Impute specific columns with MICE
missing-impute impute data.csv -o filled.csv --method mice --columns "pain,fatigue,nausea"

# Works with the demo dataset out of the box (auto-detects FACT-E columns)
missing-impute impute demo.csv -o filled.csv
```

### `missing-impute demo`

Generate a synthetic dataset for testing. Uses the FACT-E questionnaire structure (44 ordinal 0–4 items across multiple visits).

```bash
missing-impute demo -o OUTPUT [--patients N] [--visits N] [--missing RATE]
```

| Argument | Description |
|---|---|
| `-o`, `--output` | **Required.** Output CSV path |
| `--patients` | Number of synthetic patients (default: 60) |
| `--visits` | Visits per patient (default: 5, max: 10) |
| `--missing` | Missing-value rate, 0.0–1.0 (default: 0.15) |

**Example:**

```bash
missing-impute demo -o demo.csv --patients 100 --visits 6 --missing 0.2
```

### `missing-impute methods`

List available imputation methods.

```bash
missing-impute methods
```

### `missing-impute --version`

Print the package version.

```bash
missing-impute --version
```
