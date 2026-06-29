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
| `--columns` | Comma-separated column names to impute. Defaults to FACT-E items found in the file |

**Examples:**

```bash
# Impute with KNN (default)
missing-impute impute patient_data.csv -o imputed.csv

# Impute specific columns with MICE
missing-impute impute data.csv -o filled.csv --method mice --columns "gp1,gp2,gp3"
```

### `missing-impute demo`

Generate a synthetic FACT-E dataset for testing.

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
