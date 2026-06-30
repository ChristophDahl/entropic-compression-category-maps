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
│   │   ├── plot_resnet50_feature_maps_example.m
│   │   ├── run_resnet50_conditional_controls.m
│   │   ├── main_mild_balanced_supervised_readout_10class_v2.m
│   │   ├── redraw_resnet50_conditional_control_2panel_v1.m
│   │   ├── redraw_mild_balanced_readout_figures_v2.m
│   │   └── print_demo03_robustness_latex_table.m
│   ├── data_prep/
│   │   └── prepare_cifar10_variants.m
│   └── utils/
│       ├── drawManualBarLegend.m
│       ├── drawManualLineLegend.m
│       ├── labelSubplots.m
│       ├── placeLegendAt.m
│       ├── placeLegendAtAxes.m
│       ├── ec_entropy_discrete.m
│       ├── ec_joint_entropy_discrete.m
│       ├── ec_mutual_information_discrete.m
│       ├── ec_nmi_target.m
│       ├── ec_conditional_entropy.m
│       ├── ec_conditional_mutual_information.m
│       ├── ec_conditional_nmi_target.m
│       ├── ec_shuffle_within_group.m
│       ├── ec_permutation_nmi.m
│       ├── ec_permutation_conditional_nmi.m
│       └── ec_bootstrap_ci_mean.m
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
## Dataset note

The neural-network pipeline requires the CIFAR-10 MATLAB batch files, which are not bundled with this repository. Please download CIFAR-10 separately and place the extracted MATLAB files in `data/cifar-10-batches-mat/` before running the CIFAR-10 preparation and neural-analysis scripts.

## Canonical entry points

### Synthetic demonstrations
Run:

```matlab
run('src/synthetic/run_synthetic_category_map_demos.m')
```

This script generates the synthetic category-map demonstrations, summary tables, and manuscript figures.These analyses evaluate category entropy, mutual information, normalised mutual information, conditional entropy, target preservation, and robustness under perturbation for explicitly defined category maps.

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

### Qualitative feature-map figure

Run:

```matlab
run('src/neural/plot_resnet50_feature_maps_example.m')
```

This script produces the qualitative feature-map figure for one example image across nuisance variants.


### Revision analyses

The revision analyses were added to address whether unsupervised layer-derived category maps preserve object-relevant information, nuisance-condition information, or object information that remains detectable after nuisance condition is controlled.

### Strong pooled-nuisance conditional control

Run:

```matlab
run('scripts/run_revision_resnet50_controls_strong.m')
```
This script calls:

```matlab
src/revision/run_resnet50_conditional_controls.m
```
in strong pooled-nuisance mode. It evaluates object information, nuisance information, conditional object information given nuisance condition, and null-corrected conditional object information using within-nuisance permutation baselines.

### Mild nuisance-control analysis

Run:

```matlab
run('scripts/run_revision_resnet50_controls_mild.m')
```
This script applies the same conditional-analysis pipeline to a milder balanced nuisance-control image set. This analysis tests whether the dominance of nuisance information depends on the strength of the imposed transformation structure.

### Supervised-readout control

Run:

```matlab
run('scripts/run_revision_supervised_readout_mild.m')
```
This script runs the target-specific supervised readout analysis. Object-trained and nuisance-trained networks are compared to test whether the dominant recoverable information follows the specified training target.


## CIFAR-10 preparation helper

After placing the CIFAR-10 MATLAB batch files in `data/cifar-10-batches-mat/`, run:

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
