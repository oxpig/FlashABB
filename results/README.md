# Results

Per-antibody / per-sequence results referenced in the manuscript.

- **`table1_per_antibody_rmsd.csv`** -- per-antibody backbone RMSD (by CDR/framework region) for
  FlashABB and unrefined ABodyBuilder3 on the 100-structure test set, underlying Table 1 and the
  paired statistical analysis in the Appendix.
- **`inverse_folding_benchmark_results.csv`** -- full inverse-folding benchmark results (initial
  AbMPNN sequence, FlashABB-optimized sequence, recovery, per-CDR RMSD) for all 100 test
  structures.
- **`abmpnn_only_sequences.csv`** -- the plain AbMPNN-only sequences (before FlashABB-RMSD
  optimization) used as the baseline for the independent refold comparison.
- **`independent_refold_comparison.csv`** -- tidy long-format table reproducing the independent
  refold check (Table G3): RMSD of both AbMPNN-only and FlashABB-optimized sequences, refolded
  under three structure predictors (FlashABB self, Boltz-2, ABB3).

The DevAbDab dataset is not yet included here -- its exact scope (filtered from the full paired
OAS pretraining corpus, vs. the smaller held-out OAS test set) is still being confirmed.
