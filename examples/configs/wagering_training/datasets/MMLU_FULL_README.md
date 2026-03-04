# MMLU Full Dataset Configuration

## Overview

The `mmlu_full` dataset is a comprehensive configuration for the wagering pipeline that uses the **CAIS MMLU dataset** with the **"all" subset**, covering all 57 subjects available in the MMLU benchmark.

## Dataset Details

- **Source**: `cais/mmlu` (Hugging Face)
- **Subset**: `all` - includes all 57 academic subjects
- **Configuration file**: `mmlu_full.yaml`

## Available Splits

### Training
- **Split name**: `auxiliary_train` (or `train` if `auxiliary_train` is not available)
- **Purpose**: Auxiliary training data for wagering methods
- **Typical size**: ~10,000 examples

### Testing
- **Split name**: `test`
- **Purpose**: Evaluation and validation of wagering methods
- **Typical size**: ~14,000 examples

## Usage

### In Main Wagering Config

To use `mmlu_full` in your wagering pipeline training configuration, reference it in your main config file:

```yaml
datasets:
  - config_path: examples/configs/wagering_training/datasets/mmlu_full.yaml
    train_split: auxiliary_train  # or "train"
    eval_split: test
```

### Example Training Command

```bash
python scripts/wagering_train.py \
  --config examples/configs/wagering_training/my_config.yaml \
  --datasets mmlu_full \
  --output_dir workdir/mmlu_full_wagering
```

## Format Details

The dataset is automatically processed to handle the MMLU format:

- **Input format**: 
  ```
  Question: [question text]
  A) [option A]
  B) [option B]
  C) [option C]
  D) [option D]
  ```

- **Target format**: Single letter (A, B, C, or D)

- **Special handling**:
  - Converts integer answer indices (0-3) to letters (A-D)
  - Handles list-based and dict-based choice formats
  - Validates all answers are valid (A-D only)
  - Skips invalid samples with detailed logging

## YAML Configuration

```yaml
name: ['cais/mmlu', 'all']
display_name: mmlu_full
text_column: question
label_column: answer
batch_size: 5
max_prompt_tokens: 1200
load_from_disk: false
trust_remote_code: false
instruct: false
prompt: "Return the label of the correct answer for the question below.\n\nQuestion: {question}\nChoices:\n{choices}\nAnswer:"
description: ""
```

## Key Features

1. **Comprehensive Coverage**: Includes all 57 MMLU subjects
2. **Large-scale**: Provides 10,000+ training examples for robust model training
3. **Consistent Formatting**: Automatically processes raw MMLU format to match pipeline requirements
4. **Flexible Splits**: Supports both auxiliary training and test splits
5. **Error Handling**: Robust validation with detailed logging of skipped samples

## Differences from LM-Polygraph MMLU

- **Source**: Uses official CAIS MMLU (`cais/mmlu`) instead of LM-Polygraph's curated version
- **Subset**: "all" subset includes all 57 subjects vs. potentially limited subjects
- **Split names**: Uses `auxiliary_train` and `test` splits directly from CAIS

## Notes

- The dataset loading automatically detects and handles the `cais/mmlu` format
- First sample is logged for debugging purposes when loading
- Any invalid samples are skipped with warnings in the log
- Compatible with multi-LLM ensemble wagering training
