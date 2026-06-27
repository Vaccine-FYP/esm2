# Running VaxSeer

This repository contains our modified version of [VaxSeer](https://github.com/wxsh1213/vaxseer) for influenza vaccine strain selection using AI-based evolutionary and antigenicity modelling.

## Setup

### 1. Environment

```bash
conda env create -f environment.yaml
conda activate vaxseer
pip install xlrd
```

### 2. Data

The following data files are too large for GitHub. Download them from Google Drive (ALL included in vaxseer+data.zip) under the specified paths:

**Google Drive link:** https://drive.google.com/drive/folders/13cngduvl3Y4Jw37dmyutckCP4Qd6lNWs?usp=drive_link

| File | Path | Description |
|------|------|-------------|
| `ha.fasta` (224 MB) | `data/gisaid/ha.fasta` | Merged HA protein sequences (390,271 sequences) |
| `metadata.csv` (190 MB) | `data/gisaid/metadata.csv` | Merged metadata (369,295 deduplicated entries) |
| `ha_processed/` | `data/gisaid/ha_processed/` | Preprocessed training and evaluation data |
| `recommended_vaccines_from_gisaid_ha/` | `data/recommended_vaccines_from_gisaid_ha/` | HA sequences for WHO-recommended vaccine strains |

### 3. Pre-trained model checkpoints

Download from the VaxSeer Dropbox and place in `runs/`.

## Data Sources

- HA sequences and metadata are from [GISAID EpiFlu](https://gisaid.org/)
- 386 batches were scraped covering subtypes A/H3N2, A/H1N1, and others
- Date range: 2003–2023

