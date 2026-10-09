# All-layer Q/K/V checkpoint analysis

Analyzed matrices: 84

These are stored-checkpoint observations, not a controlled SFT/RLVR comparison or a GeoRA training experiment.

## Median and range across the analyzed layers

| Projection | Metric | Minimum | Median | Maximum |
|---|---|---:|---:|---:|
| Q | relative_update_rlvr | 0.0007919624 | 0.001193364 | 0.001393648 |
| Q | nss_rlvr | 2.877223e-05 | 3.277863e-05 | 4.064412e-05 |
| Q | relative_update_sft | 0.101 | 0.1913409 | 0.2427746 |
| Q | nss_sft | 0.02258648 | 0.02654198 | 0.03470435 |
| Q | min_input_overlap_k16 | 0.9999318 | 0.9999954 | 0.9999995 |
| Q | min_output_overlap_k16 | 0.9999316 | 0.999995 | 0.9999992 |
| Q | head_16_update_energy | 0.01595407 | 0.04447962 | 0.2015525 |
| Q | rank_16_retained_energy | 0.2194894 | 0.2734483 | 0.4994426 |
| Q | rank_128_retained_energy | 0.5912638 | 0.6348576 | 0.8165812 |
| Q | input_completion_energy_fraction | 2.483429e-29 | 2.949198e-29 | 3.065219e-29 |
| Q | equal_fraction_common_bf16 | 0.6657482 | 0.6997721 | 0.7332793 |
| K | relative_update_rlvr | 0.0007227587 | 0.001216135 | 0.00211332 |
| K | nss_rlvr | 3.36152e-05 | 4.126046e-05 | 7.987473e-05 |
| K | relative_update_sft | 0.0944789 | 0.1870397 | 0.255328 |
| K | nss_sft | 0.01780594 | 0.0246433 | 0.04726889 |
| K | min_input_overlap_k16 | 0.9998162 | 0.99998 | 0.999998 |
| K | min_output_overlap_k16 | 0.9998212 | 0.9999803 | 0.9999977 |
| K | head_16_update_energy | 0.01577711 | 0.03292604 | 0.07879993 |
| K | rank_16_retained_energy | 0.3574398 | 0.4361865 | 0.7145087 |
| K | rank_128_retained_energy | 0.8566838 | 0.8877312 | 0.9566484 |
| K | input_completion_energy_fraction | 0.576386 | 0.7322636 | 0.8046688 |
| K | equal_fraction_common_bf16 | 0.5853119 | 0.6854909 | 0.7923304 |
| V | relative_update_rlvr | 0.001004232 | 0.001329821 | 0.002225664 |
| V | nss_rlvr | 2.657875e-05 | 4.267793e-05 | 0.0002188818 |
| V | relative_update_sft | 0.08954552 | 0.1200684 | 0.1435135 |
| V | nss_sft | 0.03522896 | 0.05017094 | 0.06204681 |
| V | min_input_overlap_k16 | 0.9996963 | 0.9999479 | 0.9999922 |
| V | min_output_overlap_k16 | 0.9996974 | 0.9999481 | 0.9999945 |
| V | head_16_update_energy | 0.007468574 | 0.01711754 | 0.1168486 |
| V | rank_16_retained_energy | 0.3645116 | 0.5205495 | 0.9362035 |
| V | rank_128_retained_energy | 0.8490469 | 0.8943669 | 0.9873514 |
| V | input_completion_energy_fraction | 0.6412045 | 0.7656834 | 0.832637 |
| V | equal_fraction_common_bf16 | 0.5729243 | 0.6884766 | 0.7417348 |

## Metric definitions

- `relative_update_rlvr`: ||after-before||_F / ||before||_F.
- `nss_rlvr`: ||sigma_after-sigma_before||_2 / ||sigma_before||_2.
- `min_*_overlap_k16`: smallest singular value of the top-16 cross-basis overlap; values near 1 indicate nearly equal subspaces. Raw values are saved without clamping.
- `head_16_update_energy`: sum_{i<=16} ||delta v_before,i||_2^2 / ||delta||_F^2. Uniform reference is 16/input_dimension.
- `rank_r_retained_energy`: sum of the update's first r squared singular values divided by their total. The best relative Frobenius error is sqrt(1-energy).
- `input_completion_energy_fraction`: update energy outside the compact pre-training input basis. Compact K/V SVD leaves 1280 input directions unrepresented for a 256x1536 matrix; their total is computed using the residual, without choosing arbitrary individual basis vectors.
- `equal_fraction_common_bf16`: equality after rounding BOTH checkpoints to BF16; this is not a zero-gradient count.

## Files

- `metrics.csv`: one row per layer/projection.
- `details/*.json`: full spectra, overlap singular values, directional fractions, provenance, and per-matrix timing.
- `overview.png`: six comparisons across layers.
- `run.json`: versions, numeric precision, requested tensors, and completion status.

All energy fractions are within-matrix proportions; an unweighted mean across matrices is not a whole-model energy fraction.
Q and K/V have different shapes. At rank 128, Q uses 128/1536 of its maximum possible rank, while K/V use 128/256. Compare these budgets accordingly.
