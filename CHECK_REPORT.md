# Lightweight check report

## Included functionality

- Final `UltraLightDynamicStemAll6SMBStageScaleFinal` implementation.
- Required UltraLight VM-UNet backbone, SAD-Stem, GFC-Mamba/SMB, attention bridge, and Stage-Scale modules.
- ISIC NPY preparation and loading, including the formal normalization and augmentation order.
- Formal ISIC2017/ISIC2018 EMA training and minimum-validation-loss checkpoint selection.
- ISIC test evaluation with globally accumulated DSC, mIoU, sensitivity, specificity, and accuracy.
- Frozen ISIC2017-checkpoint PH2 evaluation with no PH2 training, fine-tuning, or threshold search.
- Locked YAML configurations and pinned core environment dependencies.

## Checks completed on the packaging workstation

- All Python files passed `compileall` syntax checking.
- Both YAML configurations parsed successfully and contain the locked protocol values.
- Loss, metrics, deterministic seeding, and EMA utilities imported and passed a small CPU smoke test.
- README command-line arguments were checked against the actual argument parsers.
- Core backbone/final/inheritance model files are byte-identical to the formal frozen source. The retained SAD-Stem and GFC-Mamba class ASTs are identical to the formal source definitions.
- No checkpoint, dataset array, paper file, cache, hard-coded private absolute path, credential, API key, or access token is included.

## Not executed because of the local environment

The packaging workstation has Python 3.10.9, CPU-only PyTorch 1.12.1, and does not have `timm` or `mamba_ssm`. Therefore final-model import, fresh parameter counting, random 256 x 256 forward propagation, strict checkpoint loading, and GPU training/evaluation were not executed here. `verify_model.py` is included and must be run in the pinned CUDA environment before publication. No long training or dataset evaluation was started.

## Reproducibility differences or limitations found

1. The paper follows the original UltraLight VM-UNet preparation procedure: unsorted filesystem `glob` order followed by fixed positional boundaries (1250/150/600 and 1815/259/520). `prepare_isic.py` now uses this legacy procedure by default and records the resulting IDs in `split_manifest.csv`. Because the historical run did not archive its ID-level manifest, byte-for-byte identity with that machine's filesystem order cannot be independently proven. A filename-sorted alternative is available only through the explicit `--split-mode sorted` option and is labelled as a different ordering.
2. The original ISIC preparation used bilinear interpolation for segmentation masks. This behavior is unusual but is retained because real experiment code has priority.
3. PH2 normalization is computed from the complete 200-image PH2 evaluation array, followed by per-image min-max scaling. It does not reuse ISIC2017 normalization statistics; this matches the audited PH2 implementation.
4. The formal main protocol uses EMA decay 0.999 and selects `ema_best_val_loss.pth` solely by minimum validation loss. The cleaned code now uses the same checkpoint name and policy.

## Data and licensing issues that remain unresolved

- The repository reproduces the documented UltraLight VM-UNet split method and records all generated IDs. Depositing the historical manifest, if later recovered, would remove the remaining filesystem-order ambiguity.
- PH2 explicitly prohibits redistribution and commercial use. The repository can provide only the official download link and preparation code unless written permission is obtained. Confirm with the PAA editor whether this satisfies the mandatory data-access requirement.
- The local UltraLight VM-UNet upstream snapshot contains an MIT license. The referenced PSMamba repository exposes no license file in the audited material. Permission/licensing for public redistribution of the adapted GFC-Mamba code must be resolved.
- No project-level open-source license should be added until the PSMamba-derived code status and the authors' preferred license are confirmed.

## Publication readiness

The project structure and executable workflow are ready to upload to GitHub. Before describing the repository as a fully licensed open-source release, the authors should: (1) run `verify_model.py` in the pinned CUDA environment, (2) retain and publish the generated split manifests where dataset terms allow, and (3) resolve the PSMamba-derived code license and add an appropriate repository license.
