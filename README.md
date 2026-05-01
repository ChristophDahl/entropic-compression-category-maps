# Entropic Compression and Target Preservation in Category Maps

MATLAB code for the synthetic demonstrations and neural-network analyses used in the *Entropic Compression and Target Preservation in Category Maps* project.

## Repository layout

```text
entropic-compression-github/
├── src/
│   ├── synthetic/
│   │   └── run_synthetic_category_map_demos.m
│   ├── neural/
│   │   ├── run_resnet50_two_scenario_analysis.m
│   │   ├── plot_resnet50_saved_results.m
│   │   └── plot_resnet50_feature_maps_example.m
│   ├── data_prep/
│   │   └── prepare_cifar10_variants.m
│   └── utils/
│       ├── drawManualBarLegend.m
│       ├── drawManualLineLegend.m
│       ├── labelSubplots.m
│       ├── placeLegendAt.m
│       └── placeLegendAtAxes.m
├── data/        # input data live here (ignored by Git except placeholders)
├── results/     # generated tables and .mat files live here
├── figures/     # generated figure files live here
├── docs/
├── legacy/
│   └── original_scripts/
├── .gitignore
├── LICENSE
└── README.md
```

## Canonical entry points

### Synthetic demonstrations
Run:

```matlab
run('src/synthetic/run_synthetic_category_map_demos.m')
```

This script generates the synthetic category-map demonstrations, summary tables, and manuscript figures.

### ResNet-50 two-scenario analysis
Run:

```matlab
run('src/neural/run_resnet50_two_scenario_analysis.m')
```

This script runs the full ResNet-50 pipeline for the clean-only and pooled-nuisance scenarios. It expects CIFAR-10 batches in `data/cifar-10-batches-mat/` and writes outputs to `results/` and `figures/`.

### Plot-only regeneration from saved results
Run:

```matlab
run('src/neural/plot_resnet50_saved_results.m')
```

This script regenerates the manuscript-ready ResNet-50 figures from previously saved CSV results.

### CIFAR-10 preparation helper
Run:

```matlab
run('src/data_prep/prepare_cifar10_variants.m')
```

This helper prepares clean and nuisance-variant CIFAR-10 images and label tables.

### Qualitative feature-map figure
Run:

```matlab
run('src/neural/plot_resnet50_feature_maps_example.m')
```

This script produces the qualitative feature-map figure for one example image across nuisance variants.

## Dependencies

- MATLAB R2023b or newer recommended
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox
- Deep Learning Toolbox Model for ResNet-50 Network support package
- CIFAR-10 MATLAB batch files for the neural-network workflow

## Notes on naming and provenance

The `src/` directory contains the cleaned, canonical script names intended for reuse and public release. The original versioned scripts supplied during development are preserved unchanged in `legacy/original_scripts/` for traceability.

## Suggested GitHub workflow

1. Add input data under `data/`.
2. Run the synthetic or neural entry-point scripts from the repository root.
3. Commit only source code, documentation, and lightweight configuration files.
4. Do **not** commit generated figures, result tables, large datasets, or support-package files.

## Citation

If you use this repository in academic work, cite the associated manuscript and reference this code repository in the methods or data/code availability statement.
