# Clean Accuracy Analysis

## Main Finding

The clean accuracy drop is not a smooth round-by-round degradation. It appears as a set of sharp drops when new adversarial clients inject backdoor updates into the global model.

## Trend Summary

Example clean accuracy pattern:

```text
Epoch 11: 97.30% (normal training, no backdoor update)
Epoch 12: 38.57% (first adversarial update, large drop)
Epoch 13: 50.45% (partial recovery)
Epoch 14: 73.81% (second adversarial update)
Epoch 15: 83.00% (recovery)
Epoch 16: 73.93% (third adversarial update)
Epoch 17: 85.25% (recovery)
Epoch 18: 90.27% (fourth adversarial update, smaller impact)
Epoch 19: 93.00% (continued recovery)
Epoch 30: 96.83% (close to recovered)
Epoch 31: 96.84% (stable high accuracy)
```

## Mechanism

In the attack phase, adversarial clients compute gradients from triggered samples and push the model toward the target class. During aggregation, these updates can degrade global clean accuracy. Afterward, benign clients continue contributing clean gradients, and robust aggregation can reduce the influence of suspicious updates, allowing the global model to recover.

When another adversarial client begins poisoning, a new harmful update direction is introduced and clean accuracy can drop again.

## Metrics

| Metric | Meaning | Source file | Extraction |
| --- | --- | --- | --- |
| Clean Accuracy | Main-task accuracy on clean test data | `test_result.csv` | Global row per epoch |
| ASR | Backdoor attack success rate | `posiontest_result.csv` | Global row per epoch |
| Trigger ASR | Success rate under a specific trigger | `poisontriggertest_result.csv` | Global row per epoch |

## Using `gc.py`

The `gc.py` script summarizes result folders matching `alpha_*`. It extracts clean accuracy and ASR from CSV files, saves a summary table, and plots ACC/ASR versus the alpha value.

## Conclusion

The observed behavior is consistent with multi-stage backdoor injection: clean accuracy drops at adversarial injection rounds and then recovers as benign training updates dominate subsequent aggregation rounds.
