# Green AI: Accuracy vs. Energy Trade-offs in Model Compression

Measuring how **pruning**, **post-training INT8 quantization**, and a **lightweight architecture** trade accuracy against runtime and estimated energy when training and running image classifiers on CIFAR-100. Energy and CO₂ are estimated with [CodeCarbon](https://github.com/mlco2/codecarbon).

> ⚠️ Search this file for `TODO` and resolve every one before publishing.

## Research questions
1. What are the accuracy, runtime, and energy of ResNet-50 and MobileNetV2 on CIFAR-100?
2. Does pruning (30 / 50 / 70%) reduce energy or model size, and at what accuracy cost?
3. How does post-training INT8 quantization change model size and accuracy?
4. How does a lightweight architecture (MobileNetV2) compare to compressing ResNet-50?

## Setup
```bash
git clone https://github.com/longhn-0108/green-ai-model-compression.git
cd green-ai-model-compression
python -m venv venv && source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```
CIFAR-100 is downloaded by torchvision; it is not stored in this repo.

## Reproduce
> ⚠️ TODO: replace with the real commands and arguments of `main.py` (run `python main.py --help`).
```bash
python main.py ...   # ResNet-50 baseline
python main.py ...   # ResNet-50 pruning (0.3 / 0.5 / 0.7)
python main.py ...   # MobileNetV2 baseline
python main.py ...   # INT8 quantization
```
Every run appends one row to `results/results.csv` (columns: timestamp, model_name, technique, pruning_amount, best_epoch, val_accuracy, test_accuracy, total_time_min, total_energy_kwh, model_size_mb). All tables below are taken from that file.

## Experimental setup
- **Dataset:** CIFAR-100. ⚠️ TODO: train/val/test split sizes and whether the split is seeded and fixed across runs.
- **Metrics:** test accuracy at the best-validation epoch; total wall-clock time of the run; energy estimated by CodeCarbon; size of the saved model file (MB).
- **Two environments:** a local CPU machine and a Kaggle GPU. The log has no device column, so the device is identified by runtime (about 2,600 minutes for CPU runs vs. about 75 minutes for GPU runs). ⚠️ TODO: add a `device` column to the logger and list the exact hardware.
- **Hardware differs, so CPU and GPU rows measure different machines, not different methods.** Method comparisons are made within one device only.

## Results

### GPU (Kaggle): primary comparison
ResNet-50 baseline is the reference. Deltas are relative to it.

| Model | Method | Test acc. (%) | Δ acc. (pt) | Energy (kWh) | Δ energy | Time (min) | Best epoch | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| ResNet-50 | Baseline | 74.72 | n/a | 0.077634 | n/a | 74.55 | 50 | 90.73 |
| ResNet-50 | Pruning 30% | 74.89 | +0.17 | 0.077607 | −0.03% | 74.52 | 2 | 90.73 |
| ResNet-50 | Pruning 50% | 75.04 | +0.32 | 0.077600 | −0.04% | 74.51 | 1 | 90.73 |
| ResNet-50 | Pruning 70% | 74.78 | +0.06 | 0.077867 | +0.30% | 74.77 | 1 | 90.73 |
| MobileNetV2 | Baseline | 71.98 | −2.74 | 0.019623 | −74.7% | 18.84 | 45 | 9.21 |

### CPU (local): secondary
| Model | Method | Test acc. (%) | Energy (kWh) | Time (min) | Best epoch | Size (MB) |
|---|---|---|---|---|---|---|
| ResNet-50 | Baseline (under-trained, see notes) | 48.95 | 5.723379 | 2641.57 | 18 | 90.73 |
| ResNet-50 | Pruning 30% | 70.00 | 5.707134 | 2634.08 | 39 | 90.73 |
| ResNet-50 | Pruning 50% | 75.04 | 5.702441 | 2631.55 | 1 | 90.73 |
| ResNet-50 | Pruning 70% | 74.78 | 5.698115 | 2629.12 | 1 | 90.73 |
| MobileNetV2 | Baseline | 71.55 | 1.447343 | 667.57 | 37 | 9.21 |

The CPU ResNet-50 baseline (48.95%, best epoch 18) is not a valid reference: it is far below the GPU baseline (74.72%), so accuracy gains of the pruned CPU runs over it are not a pruning effect. ⚠️ TODO: explain why (different training length / early stop) or re-run it.

### Post-training INT8 quantization (ResNet-50)
> ⚠️ TODO: re-run quantization through the same logging pipeline and commit the script, then fill this table. Do not publish numbers that were not produced by a logged run.

| Method | Test acc. (%) | Δ acc. vs GPU baseline (pt) | Model size (MB) | Compression | Device | Measured how |
|---|---|---|---|---|---|---|
| INT8 PTQ | ⚠️ TODO | ⚠️ TODO | ⚠️ TODO | ⚠️ TODO | ⚠️ TODO | ⚠️ TODO |

Quantization is applied after training, so its time and energy cover only conversion and evaluation. They are **not comparable** to the training-run energy in the tables above and should not be put in the same column.

## Findings
1. **A lightweight architecture is the effective lever.** On the same GPU, MobileNetV2 uses 74.7% less energy (about 4.0× less) and 74.7% less time than ResNet-50, at 2.74 points lower test accuracy, with a 9.9× smaller model file (9.21 vs 90.73 MB). The same 74.7% reduction appears on CPU.
2. **Pruning at 30–70% sparsity did not make the model cheaper.** The saved model size stays at 90.73 MB, and runtime and energy change by less than 0.5% (GPU: −0.04% to +0.30%; CPU: −0.28% to −0.44%). Accuracy stays within +0.06 to +0.32 points of the GPU baseline, which is within what a single run can show. This is consistent with mask-based (unstructured) pruning, where zeroed weights are still stored and computed densely. ⚠️ TODO: confirm the pruning implementation in `src/`. Sparsity alone yields no efficiency gain without structured removal of channels or sparse kernels.
3. **Hardware dominates.** The same model uses about 98.6% less energy and about 35× less time on the GPU than on the CPU (ResNet-50: 5.723 vs 0.078 kWh). This compares machines, not methods.

## Limitations and known issues
- **Energy is an estimate that tracks runtime.** In every logged run, energy divided by time gives a constant average power: about 130 W for CPU runs and about 62.5 W for GPU runs (within 0.1%). CodeCarbon is therefore using fixed hardware-power estimates, so energy differences here follow runtime differences rather than measured power draw.
- **Validation vs. test gap in pruned runs.** Pruned ResNet-50 runs report validation accuracy of 94.5–95.1% but test accuracy of 74.8–75.0%, while the baselines show gaps below 1 point. The best epoch is 1–2 for these runs. ⚠️ TODO: explain this (for example, a train/validation split that differs between the baseline and pruning runs) and fix or re-run. Reported test accuracy is the number to rely on.
- Single run per configuration, no seeds or confidence intervals. Differences under about 1 point should not be read as real.
- Pruned CPU and GPU runs at 50% and 70% report identical accuracy to four decimals. ⚠️ TODO: state whether these are independent runs or the same checkpoint evaluated on both machines.

## Repository structure
```
src/           models, pruning, quantization, energy tracking
results/       results.csv (raw log) and figures
visual.ipynb   plots of accuracy vs. energy from results.csv
main.py        entry point
```

## References
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), arXiv:1512.03385
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, arXiv:1801.04381
- CodeCarbon: https://github.com/mlco2/codecarbon

## License
MIT
