# Green AI: Model Compression on CIFAR-100

A reproducible study of the accuracy–efficiency trade-offs of **unstructured pruning**, **post-training INT8 quantization**, and a **lightweight architecture** for image classification on CIFAR-100.

Energy consumption is estimated with [CodeCarbon](https://github.com/mlco2/codecarbon).

## Research questions

1. How do ResNet-50 and MobileNetV2 compare in accuracy, runtime, and estimated energy?
2. Does L1 unstructured pruning at 30%, 50%, and 70% sparsity reduce model size, runtime, or energy?
3. What is the accuracy and size impact of post-training INT8 quantization?
4. How does a lightweight architecture compare with compressing a larger model?

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── engine.py
│   ├── models.py
│   └── utils.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── results/
│   ├── experiment_log.csv
│   └── README.md
├── notebooks/
│   └── visual.ipynb
├── docs/
│   └── methodology.md
└── tests/
    └── test_smoke.py
```

Large checkpoints and downloaded CIFAR-100 data are intentionally excluded from the portfolio repository. They can be regenerated locally and are ignored by Git.

## Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

CIFAR-100 is downloaded automatically by `torchvision` into `./data` on the first run.

## Run experiments

```bash
# ResNet-50 baseline
python main.py --model resnet50 --technique baseline

# ResNet-50 pruning
python main.py --model resnet50 --technique pruning --pruning_amount 0.3
python main.py --model resnet50 --technique pruning --pruning_amount 0.5
python main.py --model resnet50 --technique pruning --pruning_amount 0.7

# MobileNetV2 baseline
python main.py --model mobilenet_v2 --technique baseline

# Post-training quantization
python main.py --model resnet50 --technique quantization
```

The code uses a fixed seed (`42`) and a deterministic 90/10 train/validation split. Quantization is evaluated on CPU using the `fbgemm` backend.

## Logged results

The repository contains the original experiment log in `results/experiment_log.csv`. These are **historical single-run measurements**, not confidence intervals.

### GPU runs

| Model | Method | Test accuracy | Energy (kWh) | Time (min) | Size (MB) |
|---|---|---:|---:|---:|---:|
| ResNet-50 | Baseline | 74.72% | 0.077634 | 74.55 | 90.73 |
| ResNet-50 | Pruning 30% | 74.89% | 0.077607 | 74.52 | 90.73 |
| ResNet-50 | Pruning 50% | 75.04% | 0.077600 | 74.51 | 90.73 |
| ResNet-50 | Pruning 70% | 74.78% | 0.077867 | 74.77 | 90.73 |
| MobileNetV2 | Baseline | 71.98% | 0.019623 | 18.84 | 9.21 |

### CPU runs

| Model | Method | Test accuracy | Energy (kWh) | Time (min) | Size (MB) |
|---|---|---:|---:|---:|---:|
| ResNet-50 | Baseline | 48.95% | 5.723379 | 2641.57 | 90.73 |
| ResNet-50 | Pruning 30% | 70.00% | 5.707134 | 2634.08 | 90.73 |
| ResNet-50 | Pruning 50% | 75.04% | 5.702441 | 2631.55 | 90.73 |
| ResNet-50 | Pruning 70% | 74.78% | 5.698115 | 2629.12 | 90.73 |
| MobileNetV2 | Baseline | 71.55% | 1.447343 | 667.57 | 9.21 |

### Post-training INT8 quantization

A logged ResNet-50 quantization run reports **74.05% test accuracy**, **23.50 MB** saved model size, **2.50 minutes** runtime, and **0.0012 kWh** estimated energy. This run measures conversion + evaluation and should not be interpreted as directly comparable with full training energy.

## Interpretation and limitations

- MobileNetV2 is substantially smaller and faster than ResNet-50 in the logged runs, with lower test accuracy.
- The pruning implementation is **unstructured L1 pruning**. It zeros weights but does not remove channels or change dense tensor dimensions. Therefore the saved checkpoint remains approximately the same size and ordinary dense execution does not automatically become faster.
- The pruned runs show validation accuracy around 94.5–95.1% but test accuracy around 74.8–75.0%, while the baseline validation/test gap is much smaller. This discrepancy should be investigated before making strong claims about pruning quality.
- Energy is an estimate from CodeCarbon. The logged values closely track runtime, so small energy differences should not be treated as precise measured power savings.
- The experiments are single runs without confidence intervals. Differences of roughly one percentage point or less should be interpreted cautiously.
- Historical runs were performed on different hardware environments (local CPU and Kaggle GPU). Cross-hardware comparisons describe system-level differences, not isolated algorithmic effects.

## Portfolio scope

This repository intentionally keeps the **code, experiment log, methodology, and visualization workflow**, while excluding large checkpoints and downloaded datasets. This makes the project easier to clone, inspect, and reproduce without committing hundreds of megabytes of generated artifacts.

## References

- He et al., *Deep Residual Learning for Image Recognition*, arXiv:1512.03385.
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, arXiv:1801.04381.
- CodeCarbon documentation: https://github.com/mlco2/codecarbon

## License

MIT License.
