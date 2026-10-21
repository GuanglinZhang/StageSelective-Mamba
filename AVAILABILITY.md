# Availability statement drafts

## Code availability

The source code, environment specification, model configuration, data-preparation utilities, and training and evaluation scripts for this study are available at <https://github.com/GuanglinZhang/StageSelective-Mamba>. The repository includes the final `UltraLightDynamicStemAll6SMBStageScaleFinal` implementation and commands for ISIC2017, ISIC2018, and frozen-checkpoint PH2 evaluation. No test-set-based checkpoint or threshold selection is performed.

Use this wording only after the repository is public, the tagged release matches the submitted code, and the third-party licensing issue documented in `CHECK_REPORT.md` has been resolved.

## Data availability

ISIC2017 and ISIC2018 are available from the official ISIC Challenge data portal at <https://challenge.isic-archive.com/data/>. PH2 is available from the official Universidade do Porto PH2 page at <https://www.fc.up.pt/addi/ph2%20database.html> under its provider's terms. Raw dataset files are not redistributed in the repository. Data-preparation code and the required directory structure are provided. The PH2 provider prohibits redistribution and commercial use.

The repository reproduces the original UltraLight VM-UNet split procedure (legacy filesystem `glob` order followed by the documented fixed boundaries) and writes the resulting image IDs to a reusable manifest. The historical machine's ID-level manifest was not archived, so byte-identical reconstruction of its filesystem order cannot be independently guaranteed. This limitation should not be omitted from the resubmission.
