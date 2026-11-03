# LM_Dec_Long_SimpleHierarchical

SimpleHierarchical - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: Note: 1B-token LM-02 slightly worse than 200M (1.409 -> 1.438)

## Provenance

- Source repository: LM_2023_Simple-hierarchical-transformer
- Upstream / license: lucidrains/simple-hierarchical-transformer (MIT)

## Differences from the source ([Unified] protocol)

- dim 512 depth 6 (upstream 1024/8); LM-02 uses its own loss (CE + recon + hierarchical AR), LM-05 uses CE on answers only; ignore_index -100 so byte 0 is not ignored

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 51.16 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.670 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.419 / test 1.409 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.428 / test 1.438 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 956 MB, inference 169 MB
- Peak GPU memory at length 8192 (batch 1): train 3516 MB, inference 390 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.02 (keys removed 0.01); 4 pairs @ 2048: 0.02 (keys removed 0.03); 16 pairs @ 512: 0.01 (keys removed 0.02); 16 pairs @ 2048: 0.01 (keys removed 0.02)

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check simple_hier
python run.py lm02 simple_hier      # enwik8 is downloaded to data/ on first use
```
