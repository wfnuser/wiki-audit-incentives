# Data and provenance

`data/calibration.jsonl` and `data/evaluation.jsonl` contain 2,700 records each, extracted from the archived paired service measurements. Missing outcomes remain `null`. They contain no source-document text, model-response passages, API keys, HTTP headers, or thesis administration material.

| Field | Meaning |
| --- | --- |
| `id` | Original response identifier |
| `task` | Summary, QA, or Data2txt |
| `source_hash` | Content hash used to group related responses |
| `generator` | Generator identifier of the original RAGTruth response |
| `m` | Investigation service's returned probability estimate; null on failure |
| `s` | `int(m >= 0.5)`; null if unavailable |
| `a` | Adjudication accepted the factual challenge (1) or did not (0); null on failure |
| `verdict` | Original categorical adjudication output |
| `y` | At least one human-reference hallucination span |
| `y_strict` | At least one such span excluding annotations marked implicit_true |
| `m_status`, `a_status` | Terminal measurement status |
| `m_tokens`, `a_tokens` | Reported total tokens for each service; null if unavailable |

The two labels `y` and `y_strict` are derived from RAGTruth annotations. They evaluate factual quality; they do not define model adjudication. RAGTruth and its underlying source material remain subject to upstream terms. No upstream text corpus is redistributed here.

`contracts.json` contains contracts selected using the original assigned calibration batch, including missing-outcome completions. `protocol.json` contains the fixed service settings and six system prompts extracted from the evaluation protocol. The investigation sees the response, plus the question for QA; the adjudicator additionally sees the original source. Neither sees reference labels. Models return probability JSON or verdict JSON, respectively.

`SHA256.json` checks the public data files. `docs/provenance.json` records hashes of the original code and private archived inputs/logs used during extraction; those private files are not needed to run this release. Hashes document provenance, not independent proof that every historical model call occurred.

The release preserves the mathematical routines from the thesis. Only the archive-specific campaign loader was removed from the statistical module; the standalone entry point consumes the exported rows instead. The release does not contain live model-call runners, retries, or credentials. Reproduction means replaying measurements and statistical analysis, not regenerating them from a currently served model.

Full-assigned-sample missing-outcome bounds and complete-case estimates are both reported. The 10,000-replicate intervals in `results/reference.json` refer to the complete-case evaluation. Raw-generation reproduction and historical exploration are outside this package.
