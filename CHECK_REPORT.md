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



