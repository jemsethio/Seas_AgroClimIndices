# Seasonal Agroclimate Indices Forecast Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-operational%20workspace-green.svg)]()
[![License](https://img.shields.io/badge/license-to%20be%20added-lightgrey.svg)]()


The **Seasonal Agroclimate Indices Forecast Pipeline** is an operational Python-based workflow designed to support the seasonal implementation of agroclimate indices forecasts for agriculture, food security, drought preparedness, early warning, climate-risk management, and policy decision-making. The pippline is designed for operational climate services, enabling seamless integration into platforms such as EDACaP and the AgroClimate Africa API, and supporting early warning, advisory generation, and climate risk decision-making at scale by transforms seasonal climate forecast data into actionable agroclimate information products. It supports the full workflow from climate data acquisition to country-specific processing, bias correction, downscaling, agroclimate index computation, multi-model ensemble generation, visualization, reporting, and policy-ready outputs.

The system is designed for **multi-country implementation**. Ethiopia is currently used as the main production implementation case, while the workflow can be extended to other countries by adding country configuration files, boundary data, metadata, and country-specific output directories. 

Repository: [https://github.com/jemsethio/AgClimateAF_indices](https://github.com/jemsethio/AgClimateAF_indices)


---

## What This Repository Does

This repository provides a structured production environment for generating seasonal agroclimate forecast products. It supports:

- Seasonal forecast data download and organization
- Country-specific configuration and spatial subsetting
- Bias correction and downscaling workflows
- Agroclimate indices computation
- Multi-model ensemble generation
- Spatial and point-based forecast extraction
- Country-level forecast product generation
- Map and visualization production
- Publication-quality figures
- Policy-ready summaries and climate information products
- Smoke testing and diagnostic checks
- Scalable implementation across multiple countries

---

## Key Functionality

### 1. Seasonal Forecast Data Acquisition

The pipeline includes tools for downloading and organizing seasonal forecast data from CDS-compatible sources. Data can be processed by country, forecast initialization year, month, and model.

### 2. Bias Correction and Downscaling

The workflow supports the preparation of improved forecast products through bias correction and downscaling steps. These processes are essential for translating raw seasonal forecast data into spatially meaningful and agriculturally relevant information.

### 3. Agroclimate Indices Computation

The pipeline computes agroclimate indices that are useful for agriculture and climate-risk decision-making. These may include rainfall, temperature, drought, moisture, crop-season, and other climate-sensitive indicators depending on the configured implementation.

### 4. Multi-Model Ensemble Production

The pipeline supports generation of multi-model ensemble products to improve forecast robustness and support probabilistic interpretation of seasonal climate risks.

### 5. Visualization and Mapping

The repository includes visualization modules for producing spatial forecast maps and agroclimate index products suitable for technical review, reporting, and communication.

### 6. Policy-Ready Outputs

The publication modules support generation of figures, reports, and policy-oriented outputs that can be used by decision-makers, national institutions, and climate service partners.

### 7. Multi-Country Scalability

The system is designed around country configuration files, allowing the same workflow to be adapted for different countries by updating metadata, boundaries, and canonical paths.

---

## Repository Structure

```text
src/cds_agroclimate_pipeline/
  cds/              # Data download, index computation, MME, point extraction
  publication/      # Publication figures and policy brief generators
  visualization/    # Map products

scripts/pipeline/   # Shell orchestrators for production workflows

config/countries/   # Country metadata, bounding boxes, and canonical paths

data/
  countries/
    ethiopia/
      boundaries/
      reports/
      seasonal/cds/
    zambia/
      smoke_tests/
  regional/
    Africa/
    EAF africa/
    WAF/
  cache/

tools/diagnostics/  # Ad hoc inspection utilities

tools/smoke_tests/  # Small end-to-end smoke tests

tests/              # Unit tests

archive/            # Legacy scripts and generated build/cache artifacts
```

Country-specific products are stored under:

```text
data/countries/<country>/
```

Regional Open-Meteo products are stored under:

```text
data/regional/africa/openmeteo/
```

For Ethiopia, the canonical CDS root is:

```text
data/countries/ethiopia/seasonal/cds
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jemsethio/AgClimateAF_indices.git
cd AgClimateAF_indices
```

### 2. Create a Python Environment

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Using `uv`:

```bash
uv venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

If the repository includes a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If the repository includes a `pyproject.toml` file:

```bash
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
```

---

## CDS API Configuration

The pipeline requires access to CDS-compatible seasonal forecast data.

Ensure that your CDS API credentials are configured in:

```text
~/.cdsapirc
```

A typical CDS API configuration looks like:

```text
url: https://cds.climate.copernicus.eu/api
key: <your-cds-api-key>
```

---

## Environment Variables

The scripts default to Ethiopia, but another country can be selected using the `--country` argument or environment variables.

Example:

```bash
export AGROCLIMATE_COUNTRY=ethiopia
export AGROCLIMATE_PROJECT_ROOT=/Volumes/T7/agroclimate_data
```

The `AGROCLIMATE_PROJECT_ROOT` variable is useful when running the pipeline from an external production workspace, especially when handling large seasonal forecast datasets, raster files, and country-level products.

---

## Main Commands

### Run the Full Country Pipeline

```bash
bash scripts/pipeline/run_country_pipeline.sh \
  --country ethiopia \
  --year 2026 \
  --month 5
```

This command runs the full seasonal forecast production workflow for the selected country, year, and initialization month.

### Watch a Country Pipeline Run

```bash
bash scripts/pipeline/watch_pipeline.sh \
  --country ethiopia \
  --year 2026 \
  --month 5
```

This command monitors model outputs and triggers downstream products as data become available.

### Download Seasonal Surface Forecast Data

```bash
PYTHONPATH=src .venv/bin/python -m cds_agroclimate_pipeline.cds.download_surface \
  --country ethiopia \
  --year 2026 \
  --month 5 \
  --models ecmwf
```

### Compute Agroclimate Indices

```bash
PYTHONPATH=src .venv/bin/python -m cds_agroclimate_pipeline.cds.compute_indices \
  --country ethiopia \
  --year 2026 \
  --month 5 \
  --models ecmwf
```

### Build the Multi-Model Ensemble

```bash
PYTHONPATH=src .venv/bin/python -m cds_agroclimate_pipeline.cds.build_mme \
  --country ethiopia \
  --year 2026 \
  --month 5
```

---

## Seasonal Forecast Implementation Workflow

A typical seasonal agroclimate forecast implementation follows the sequence below:

```text
1. Define the country configuration
2. Prepare national boundary and metadata files
3. Download seasonal forecast data
4. Apply preprocessing, bias correction, and downscaling
5. Compute agroclimate indices
6. Generate model-specific forecast products
7. Build multi-model ensemble outputs
8. Extract spatial and point-based forecast information
9. Produce maps, figures, and policy-ready summaries
10. Validate outputs using diagnostics and smoke tests
11. Archive or publish final seasonal forecast products
```

This workflow supports operational production of agroclimate information for national climate services, agricultural advisory systems, drought preparedness, early warning, and policy planning.

---

## Adding a New Country

To add a new country implementation, follow the steps below.

### 1. Add a Country Configuration File

Create a new configuration file:

```text
config/countries/<country>.yaml
```

### 2. Create Required Country Directories

Create the required folder structure:

```text
data/countries/<country>/boundaries
data/countries/<country>/reports
data/countries/<country>/seasonal/cds
```

### 3. Add National Boundaries and Metadata

Country-specific boundary files, administrative layers, and configuration metadata should be placed under:

```text
data/countries/<country>/boundaries
```

### 4. Run the Pipeline

```bash
bash scripts/pipeline/run_country_pipeline.sh \
  --country <country> \
  --year 2026 \
  --month 5
```

Example:

```bash
bash scripts/pipeline/run_country_pipeline.sh \
  --country zambia \
  --year 2026 \
  --month 5
```

The core data pipeline is country-configurable. However, some publication and policy narrative components under:

```text
src/cds_agroclimate_pipeline/publication/
```

are currently Ethiopia- and Ministry of Agriculture-focused and may require adaptation before use in other country contexts.

---

## Testing

Run unit tests with:

```bash
PYTHONPATH=src:agroclimate_indices .venv/bin/python -m pytest tests
```

Smoke tests are available under:

```text
tools/smoke_tests/
```

Smoke tests should be used to validate small end-to-end workflows before launching full production runs.

---

## Diagnostics

Diagnostic tools are available under:

```text
tools/diagnostics/
```

These tools support inspection of downloaded forecast data, intermediate files, agroclimate indices, ensemble products, and map outputs.

---

## Data Management

The repository separates country-level, regional, cache, and legacy outputs.

Country-level products:

```text
data/countries/<country>/
```

Regional products:

```text
data/regional/
```

Cache and temporary products:

```text
data/cache/
```

Legacy scripts and generated artifacts:

```text
archive/
```

Large datasets should not be committed to Git unless they are small reference files required for testing or reproducibility.

Recommended practice:

- Keep country configuration files under version control.
- Store large forecast datasets on external or institutional storage.
- Use consistent country slugs across configuration files, data folders, and scripts.
- Separate raw forecasts, processed indices, ensemble outputs, maps, and final reporting products.
- Archive final seasonal products in a clearly documented country-specific folder.
- Avoid committing generated data, cache files, and large raster products to Git.

---

## Example: Ethiopia Seasonal Forecast Production

```bash
export AGROCLIMATE_COUNTRY=ethiopia
export AGROCLIMATE_PROJECT_ROOT=/Volumes/T7/agroclimate_data

bash scripts/pipeline/run_country_pipeline.sh \
  --country ethiopia \
  --year 2026 \
  --month 5
```

The Ethiopia seasonal forecast products are organized under:

```text
data/countries/ethiopia/seasonal/cds
```

---

## Intended Users

This repository is intended for:

- Climate data scientists
- Agroclimate service developers
- National meteorological and agricultural institutions
- Agricultural advisory system developers
- Early warning and drought preparedness teams
- Policy analysts and decision-support teams
- Researchers working on seasonal forecasting, climate risk, and agricultural adaptation

---

## Outputs

Depending on the selected workflow and country configuration, the pipeline can generate:

- Downloaded seasonal forecast datasets
- Bias-corrected forecast products
- Downscaled forecast products
- Agroclimate index rasters and summaries
- Multi-model ensemble products
- Point-based forecast extractions
- Country-level maps
- Diagnostic outputs
- Publication-ready figures
- Policy-ready summaries and reports

---

## Recommended Operational Practice

For production use, it is recommended to:

1. Confirm the country configuration before running the pipeline.
2. Validate boundary files and country metadata.
3. Confirm CDS API access.
4. Run smoke tests before a full production run.
5. Document the forecast initialization year, month, country, and model list.
6. Review intermediate outputs before producing final maps and reports.
7. Validate final maps and policy products before dissemination.
8. Archive final outputs using a clear country-year-month folder structure.

---

## Author

**Jemal Ahmed**  
Email: [J.Ahmed@cgiar.org](mailto:J.Ahmed@cgiar.org)

---

## Acknowledgement

This project is supported by the **CGIAR Climate Action Science Program** of the **Alliance of Bioversity International and CIAT**. The work contributes to strengthening climate information services, agroclimate advisory systems, and policy-ready seasonal forecast products for agriculture and food systems.

---

## Citation

If you use this repository for research, operational climate services, policy analysis, or decision-support system development, please cite it as:

```text
Ahmed, J.; Ghosh, A. (2026). Seasonal Agroclimate Indices Forecast Pipeline: A production workspace for multi-country seasonal agroclimate indices forecasts and policy-ready outputs. GitHub repository: https://github.com/jemsethio/AgClimateAF_indices
```

---

