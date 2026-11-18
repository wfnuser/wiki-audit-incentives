# Wiki Audit Incentives

**Incentives for Crowdsourced Wiki Auditing** · Wiki 众包审核激励机制

How can a platform reward contributors for investigating factual errors without making uninformed submissions profitable? This repository studies reward–deposit contracts when the platform's adjudication can itself be wrong.

Wiki maintenance is the motivating application. The experiments use **RAGTruth** responses in summarization, question answering, and data-to-text generation as factual-auditing tasks. They are offline measurements and contract evaluations, not a deployment in a live Wiki community.

This repository provides the offline contract solver, numerical measurements, statistical replay, and regression tests. It contains no manuscript or presentation files.

## Quick start

Python 3.10+ and NumPy are required. No API key, GPU, or new model calls are needed.

```bash
git clone https://github.com/wfnuser/wiki-audit-incentives.git
cd wiki-audit-incentives
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Recalibrate contracts and replay counts, utilities, and token usage.
python reproduce.py --output results/local-replay.json

# Also reproduce the 10,000-replicate source-cluster intervals.
python reproduce.py --bootstrap 10000 --output results/local-bootstrap.json
```

Outputs are written only to a new path; existing results are never overwritten. The checked-in [reference output](results/reference.json) includes the 10,000-replicate run. [Data checksums](data/SHA256.json) are verified automatically before analysis.

## Model

A contributor can investigate at cost `c`, observe a binary signal `S`, and decide whether to submit a challenge. The platform accepts (`A=1`) or rejects (`A=0`) the submitted challenge. An accepted challenge earns reward `R`; a rejected challenge forfeits deposit `D`. The calculations here use submission fee `f=0` and normalized investigation cost `c=1`.

Let `q_ij = P(S=i, A=j)`. Two relevant expected utilities are:

```text
Blind submission:             U_B = (q_01 + q_11) R - (q_00 + q_10) D
Investigate; submit if S=1:    U_I = -c + q_11 R - q_10 D
```

We seek contracts that discourage blind submissions and make investigation worthwhile. All six pure policies are evaluated: exit; submit without investigation; investigate and never submit, always submit, submit only for a positive signal, or submit only for a negative signal. The signal indicates suspicion; it is not ground truth.

The solver minimizes the reward subject to utility margins and computes a feasible deposit interval. The archived protocol rounds the minimum reward upward to an integer credit and chooses the midpoint of that interval. Credits are normalized experimental utility units, not currency.

For robust contracts, the same `R,D` must satisfy the constraints for every probability table in a total-variation neighborhood:

```text
TV(q, q_hat) = 0.5 * sum_ij |q_ij - q_hat_ij| <= radius
```

The default margins are `U_B <= -0.01` and `U_I >= 0.05`. A radius of `0.01` describes uncertainty in the probability table; it is distinct from those utility margins and is not a 95% confidence radius. Exact rational arithmetic and worst-case distributions are used in the contract solver; a general-purpose LP package is not required.

## Experimental protocol

| Batch | Assigned responses | Source documents | Complete investigation–adjudication pairs |
| --- | ---: | ---: | ---: |
| Calibration | 2,700 | 1,350 | 2,686 |
| Independent evaluation | 2,700 | 450 | 2,688 |

Source-content hashes are disjoint across batches. Each batch includes Summary, QA, and Data2txt tasks; no model weights are trained.

The investigation call sees the model response (and the question for QA), **without the original source**, and returns `p_accept`: its estimate that full-source adjudication would accept a factual challenge. The fixed decision rule is `p_accept >= 0.5`. The adjudication call separately reads the complete source and response; only the `contradicted` verdict counts as acceptance. The human reference labels are used to evaluate quality, not to settle contracts.

Both services were measured for each assigned response, then alternative routing policies were replayed offline. Thus “fewer reviews” refers to the reconstructed policy, not to fewer calls made during data collection. [Protocol settings and the six system prompts](data/protocol.json) record the service identifier `qwen3.8-flash`, temperature `0.2`, and the other archived settings. The service identifier is not a downloadable model snapshot.

### Missing measurements and analysis populations

Contract calibration retains all 2,700 assigned calibration responses and considers every attainable completion of missing binary measurements. The primary frozen contract is `R=14, D≈7.9894`; the predefined TV-radius-0.01 contract is `R=20, D≈11.8127`. Neither is refitted on the evaluation batch.

The defense presentation additionally reports **complete-case** evaluation on 2,688 responses. This repository reproduces that table and its intervals, while also reporting assigned-sample utility bounds with the missing outcomes retained. These are different analysis populations. The complete-case numbers do not replace the original missing-outcome bounds or assume that missingness is random.

## Main complete-case results

| Evaluation policy | Sent for adjudication | Accepted true errors | Accepted false alarms |
| --- | ---: | ---: | ---: |
| Full adjudication | 2,688 | 657 | 240 |
| Investigate; submit suspected items | 1,966 | 621 | 202 |
| Task-matched random submission (expectation) | 1,966 | 556.65 | 163.41 |

The investigation-based routing uses **26.9% fewer reviews** and retains **94.5%** of the true errors identifiable by full adjudication. Relative to random submission with the same per-task review counts, it finds more true errors but also produces more false alarms. A false alarm means the platform accepts a challenge against a response with no positive human-reference error label.

| Frozen contract | Mean net credit per evaluation response | Approx. 95% source-cluster interval |
| --- | ---: | ---: |
| Primary, `R=14` | −0.111 | [−0.441, 0.232] |
| TV-robust, `R=20` | +0.100 | [−0.378, 0.596] |

Intervals use 10,000 task-stratified source-cluster bootstrap replicates, seed `20260917086`. Each replicate draws 150 source groups with replacement within each of the three tasks, retaining all complete responses of each selected source. It does **not** sample just 450 individual responses. Both intervals cross zero: a positive sample mean does not establish reliably positive population utility or individual profitability.

Token replay uses 2,421,340 tokens for full adjudication, versus 897,579 for investigation plus 1,807,315 for selected adjudication: **11.71% more total tokens**. Saving review slots is not the same as saving total tokens or money. Different contributor/platform prices may change monetary comparisons.

## Repository map

| Path | Purpose |
| --- | --- |
| `reproduce.py` | Offline entry point: hashes, contract reconstruction, policy replay, and bootstrap |
| `experiments/large_sample_statistics.py` | Missing-outcome completions, contract calibration, six-policy utilities, cluster bootstrap |
| `experiments/calibration_robustness.py` | Total-variation extrema and uncertainty calculations |
| `experiments/incentive_uncertainty.py` | Utility and interval helpers |
| `tests/` | Existing mathematical and statistical regression tests |
| `data/` | Numeric measurement rows, frozen contracts, prompts, settings, checksums |
| `results/reference.json` | Reproduced reference results |
| `docs/DATA.md` | Data schema, provenance, and scope of release |
| `docs/provenance.json` | Original-source hashes and packaging adaptations |

## Interpretation and scope

The model assumes risk-neutral expected utility, a specified strategy menu, and a fixed adjudication and settlement rule. Robust feasibility applies within the chosen uncertainty set. It does not guarantee that every contributor profits, that arbitrary extra strategies remain unprofitable, or that the platform honestly honors payments without additional enforcement.

This release supports numerical contract calibration and offline resource/quality comparisons. It does not establish a causal change in autonomous contributors' willingness to investigate. A prospective comparison against direct assignment or payment for verifiable investigation remains separate work. Historical exploratory experiments and the small evidence-location probe are outside this core release.

## Data attribution

RAGTruth: Cheng Niu et al., **RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models**, ACL 2024, pp. 10862–10878. DOI: [10.18653/v1/2024.acl-long.585](https://doi.org/10.18653/v1/2024.acl-long.585).

The archived dataset commit is `c103204b9ce28d6bbad859304bf30de72b8ed8fe`. This repository contains derived measurement tables and reference indicators, not original source documents or generated passages. Upstream data retains its original rights; see [data documentation](docs/DATA.md).

## License

Code is released under the [MIT License](LICENSE). That license does not relicense RAGTruth or its underlying source material.
