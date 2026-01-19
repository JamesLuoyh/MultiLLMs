# Cache Separation: Training vs Evaluation

## Summary

✅ **Confirmed**: Training and evaluation phases use **DIFFERENT cache keys** and do **NOT** share cache.

## Evidence from Test Logs

### Training Phase Cache (Size 20)
```
2026-01-14 16:24:01,314 - Combined datasets: 20 examples (not shuffled yet)
2026-01-14 16:24:01,320 - Cache hit for models [...] and dataset size 20
2026-01-14 16:24:01,320 - Using cached training logits from shared cache
2026-01-14 16:24:01,323 - Cache hit for models [...] and dataset size 20
2026-01-14 16:24:01,324 - Using cached hidden states from shared cache
```

### Evaluation Phase Cache (Size 10)
```
2026-01-14 16:24:25,686 - Cache hit for models [...] and dataset size 10
2026-01-14 16:24:25,687 - Using cached logits from shared cache
2026-01-14 16:24:25,866 - Cache hit for models [...] and dataset size 10
2026-01-14 16:24:25,880 - Cache hit for models [...] and dataset size 10
```

## Cache Key Generation

The cache key is computed as:
```python
cache_key = (
    model_paths,           # Same for training and evaluation
    (dataset_size, hash),  # DIFFERENT: 20 vs 10
    option_tokens          # Same for training and evaluation
)
```

### Training Dataset
- **Dataset**: Combined training datasets
  - mmlu train split: 10 examples
  - medmcqa train split: 10 examples
- **Total size**: 20 examples
- **Cache key**: `(models, (20, hash_of_combined_dataset), option_tokens)`

### Evaluation Datasets
- **Datasets**: Individual test/validation datasets
  - mmlu test split: 10 examples
  - medmcqa validation split: 10 examples
  - arc_easy test split: 10 examples
- **Each dataset size**: 10 examples
- **Cache key**: `(models, (10, hash_of_test_dataset), option_tokens)`

## Key Differences

| Aspect | Training | Evaluation |
|--------|----------|------------|
| Dataset size | 20 (combined) | 10 (individual) |
| Dataset splits | train splits | test/validation splits |
| Cache key size | 20 | 10 |
| Cache files | Separate cache file | Separate cache file |

## Critical: Data Leakage Prevention

⚠️ **IMPORTANT**: Even if training and testing datasets have the **same size**, they **NEVER share cache** because:

1. **Training uses train splits** (e.g., `train` split)
2. **Testing uses test/validation splits** (e.g., `test` or `validation` split)
3. **Different splits = Different examples = Different cache keys**

### Protection Mechanism

The cache key includes a **content hash** based on the first 3 examples:
```python
dataset_signature = (dataset_size, content_hash)
content_hash = MD5_hash(first_3_examples)
```

**Why this prevents data leakage:**
- Train split has **different examples** than test split
- Different examples → **Different content hash** → **Different cache keys**
- Therefore: **Training and testing ALWAYS use separate cache files**

### Example: Same Size, Different Cache

Even if both train and test have 10 examples:
- **Train split**: `(10, hash_of_train_examples)` → Cache file A
- **Test split**: `(10, hash_of_test_examples)` → Cache file B
- **Result**: Different cache files, **NO data leakage** ✅

## Conclusion

✅ **Training and evaluation use DIFFERENT cache keys** and **NEVER share cache**, even when sizes match, because:
1. **Different dataset splits**: Training uses train splits, evaluation uses test/validation splits
2. **Different content**: Different splits have different examples → different content hash
3. **Content hash in cache key**: Ensures train and test splits always get different cache keys

This is **critical for correctness** - using training data for testing would cause serious data leakage!
