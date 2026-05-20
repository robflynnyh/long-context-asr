# ROB-98 Poor SSL Performance Report

Date: 2026-05-20

## Scope

This report investigates why the ROB-91 frozen probes over the ROB-70 BEST-RQ SSL checkpoints performed poorly. It compares the repository implementation against the open BEST-RQ implementation described in arXiv:2405.04296 and the current SpeechBrain BEST-RQ recipe.

Required preflight was completed before planning or editing:

- `symphony/instructions/linear-context.md`: required a recent Linear comment reread. The initial report was written when ROB-98 had no comments; a later human clarification asked where the 60% masking claim came from, and this revision answers that by separating the paper's start-frame mask percentages from actual masked-frame percentages.
- `symphony/instructions/repository.md`: report artifacts should live in the repo, with concise diary entries for future agents.
- `symphony/instructions/work-loop.md`: no `Branch/ref` was supplied, so this work branches from `dev`.
- `symphony/instructions/experiment-execution.md`: do not launch long GPU work unless requested. This issue asks for investigation/reporting, not a new run.
- `symphony/instructions/validation-and-handoff.md`: documentation-only validation is `git diff --check`, followed by commit, push, PR, and Linear handoff.

## Sources Inspected

External references:

- Paper HTML: https://arxiv.org/html/2405.04296
- Paper PDF: https://arxiv.org/pdf/2405.04296
- SpeechBrain BEST-RQ recipe: https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/self-supervised-learning/BEST-RQ/train.py
- SpeechBrain BEST-RQ helper: https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/BESTRQ.py
- SpeechBrain BEST-RQ hparams: https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/self-supervised-learning/BEST-RQ/hparams/BEST-RQ.yaml

Repository evidence:

- `lcasr/models/BestRQ.py`
- `exp/train_bestRQ.py`
- `exp/configs/ssl/bestrq_6l_2048_spotify.yaml`
- `lcasr/models/ctc_probe.py`
- `lcasr/models/sconformer_xl.py`
- `symphony/scripts/prepare_rob91_probe_configs.py`
- `symphony/jobs/rob91_mimas_probe_suite.sh`
- `symphony/RESEARCH_DIARY.md`
- ROB-91 output summary: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/rob91-full-bilstm-10epoch-constant-20260519T0911Z/OUTCOME.md`

## Bottom Line

The strongest current explanation is not that random-codebook BEST-RQ cannot work. The open implementation shows useful downstream ASR performance with fixed random projection/codebook targets. The poor ROB-91 probes are more plausibly explained by a combination of:

1. A major SSL masking mismatch: our current config masks about 10% of stacked frames, while the paper's successful settings are much higher: Table 3's best ablation uses 12% mask starts, about 48% actual masked frames, and the main SpeechBrain/open setting uses 15% mask starts, about 60% actual masked frames.
2. A substantially non-like-for-like SSL setup: different encoder depth, subsampling, self-conditioning, optimizer, scheduler, normalization, and training horizon.
3. Undertraining or at least weak evidence of completed pretraining: the ROB-70 checkpoint labels count consumed recordings, not optimizer steps, and the run is only one epoch.
4. Probe limitations: the final ROB-91 BiLSTM probe fixed the most obvious "linear head is too weak" problem, but it still does not match the paper's hidden-state weighted-sum probing setup and it evaluates on TEDLIUM without LM support.

There is no obvious evidence that the quantizer target path is fundamentally broken after the ROB-70 fixes. The implementation uses a fixed random projection quantizer, predicts codebook classes with cross entropy on masked positions, skips empty-mask chunks, and saves the acoustic-model state for downstream loading. The main concern is that the task being learned is easier and less representation-forcing than the SpeechBrain/paper task.

## Detailed Comparison

### 1. Masking Is The Highest-Confidence Training Mismatch

SpeechBrain/paper behavior:

- The paper's main model-settings paragraph says the open implementation increased masking to about 60% of the audio by selecting 15% of frames as mask starts and masking each selected frame plus the following three frames.
- The SpeechBrain hparams encode this final recipe as `mask_prob: 0.15` and `mask_length: 4`.
- SpeechBrain's helper explicitly defines `mask_prob` as the probability for a frame to spawn a mask, not the final masked-frame probability. The helper comment gives the example `100 * 0.15 * 4 = 60%` masked frames.
- The paper's Table 3 ablation uses the same start-frame convention. Its best row is 12% starts with codebook 8192, which corresponds to about 48% actual masked frames because each start masks four frames. The table therefore supports a high-mask-density region, but it does not itself say that 60% is the best ablation point.

Repository behavior:

- `exp/configs/ssl/bestrq_6l_2048_spotify.yaml` uses `mask_percentage: 0.1` and `frames_to_mask: 5`.
- `BestRQ.select_mask()` first groups time into non-overlapping blocks of `frames_to_mask`, samples each group with probability `mask_percentage`, then repeats the sampled decision across the group.
- Therefore the expected final stacked-frame mask fraction is approximately `mask_percentage`, not `mask_percentage * frames_to_mask`. With the current config, this is about 10% of stacked frames, not 50% or 60%.
- In SpeechBrain start-index terms, this is roughly equivalent to `mask_prob ~= 0.025` with `mask_length: 4`, or `mask_prob ~= 0.02` with a length-5 start-mask scheme. The repo's `mask_percentage: 0.1` is a group-selection probability, not the same quantity as SpeechBrain's `mask_prob: 0.12` or `0.15`.

This is a likely performance issue. The paper's ablation found masking ratio had a large downstream effect: dev-clean WER fell from roughly 35% to about 20% as the selected start-frame ratio increased from 1% to 12% with length-4 spans. Since the table percentages are start-frame percentages, that sweep moved from about 4% actual masking to about 48% actual masking. Our 10% final masking is much closer to the low-mask end than to either the best ablation row or the final open recipe.

The observed CE loss decrease from around `log(8192) = 9.01` toward about `5.7` is real learning, but it may be learning an easy local reconstruction problem. With only 10% masking, the model sees much more unmasked acoustic context around each target than it would under the paper's 48-60% actual-mask regime. That can lower the SSL loss without forcing robust, transferable hidden states.

Recommendation: before any larger rerun, add a tiny diagnostic that logs `num_masked_frames / valid_stacked_frames` distribution. Then test paper-style start-index masking with contiguous length `4` on stacked frames. The most direct targets are `mask_prob: 0.12` for the best Table 3 ablation, about 48% actual masking, and `mask_prob: 0.15` for the final SpeechBrain/open setting, about 60% actual masking. If keeping the existing grouped sampler, use `mask_percentage` near the desired final actual mask fraction, but matching the SpeechBrain start-index semantics would make comparisons cleaner.

### 2. The Quantizer Itself Looks Broadly Correct

The broad BEST-RQ mechanism matches the paper:

- Input features are 80-dimensional log-Mel-like features.
- Contiguous feature frames are stacked before quantization.
- Targets come from a fixed random projection plus fixed codebook.
- The model predicts masked-region codebook IDs with cross entropy.
- Masked input features are replaced by Gaussian noise with std `0.1`.

The `vector_quantize_pytorch.RandomProjectionQuantizer` used in `lcasr/models/BestRQ.py` registers the random projection as a buffer and calls its vector quantizer in eval mode inside forward, so it is not obviously being trained accidentally. The use of `codebook_size: 8192` and `codebook_dim: 16` also matches the original BEST-RQ codebook scale.

The recent ROB-70 empty-mask fixes also look conceptually right. Returning `loss: None` for chunks with no selected masked frames is safer than forcing one arbitrary frame, and the trainer now skips optimizer steps when no gradients exist.

Remaining risk: because `BestRQ.forward()` only reports the total masked target count, not the valid-frame denominator, it is hard to tell from W&B whether the SSL objective is operating at the intended mask density. That instrumentation should be added before interpreting more loss curves.

### 3. Architecture And Optimization Are Not Like-For-Like

Open implementation setup from arXiv:2405.04296/SpeechBrain:

- 960 hours LibriSpeech pretraining.
- 42 epochs, roughly 200k steps, with a 100k-step checkpoint.
- 8 V100 GPUs.
- Dynamic batching with about 800 seconds total batch across 8 GPUs in the paper experiment.
- 12 conformer encoder layers, `d_model: 576`, `d_ffn: 2048`.
- Two convolutional frontend layers reduce time by 4 before the transformer.
- Sentence input normalization.
- AdamW with `lr: 0.0008`, betas `(0.9, 0.98)`, weight decay `0.01`, Noam warmup `25000`.
- Paper probe uses hidden-state weighted sums for downstream tasks.

Repository ROB-70 setup:

- Spotify SSL run, one epoch.
- 6-layer `SCConformerXL`, `d_model: 768`, no dropout in the config.
- Internal subsampling factor 8.
- Self-conditioning remains enabled in the acoustic model.
- MADGRAD with LR `3e-4`, warmup `5000`, repo cosine scheduler behavior.
- Fixed 2048-frame chunks with no overlap.
- Batch size `176`, but the trainer chunks full recordings internally and saves checkpoints by consumed recording count.

Some differences may be intentional for this project, but they weaken conclusions from a direct comparison. Two details are particularly important:

- `SCConformerXL.forward(..., skip_vocab_projection=True)` still runs intermediate self-conditioning at every non-final layer when `self_conditioning` is enabled. That means the SSL encoder is not "encoder only" in the same sense as SpeechBrain's `EncoderWrapper`; it also trains and uses the model's decoder/projection path internally during SSL.
- The checkpoint names `step_25344.pt`, `step_52800.pt`, and `step_105360.pt` are recording-count checkpoints from this trainer, not direct optimizer-step equivalents to the paper's 100k/200k step checkpoints.

Recommendation: for the next diagnostic run, either disable `self_conditioning` during SSL pretraining or explicitly treat the decoder/self-conditioning stack as part of the pretrained acoustic model and make all downstream probes load it. The final ROB-91 BiLSTM setup does load the SSL decoder path, but earlier linear-probe interpretations should be considered superseded.

### 4. ROB-91 Probing Was Improved, But Still Not A Paper-Exact Probe

What ROB-91 got right by the final run:

- It moved beyond a simple linear head.
- It used a 2-layer BiLSTM with hidden size 1024 and dropout 0.2, matching the paper's ASR probe description.
- It trained for 10 epochs after the 3-epoch result was judged too short.
- It used the normal `load_model()` path and reusable `lcasr.models.ctc_probe` wrapper.
- The final 10-epoch result was consistently negative across 25%, 50%, and 100% checkpoints: 99.85%, 99.67%, and 99.73% TEDLIUM test WER.

Remaining differences:

- The paper/MP3S probing setup uses a weighted sum of hidden states from the frozen SSL model. ROB-91 uses the final hidden state only.
- The paper's ASR probe trains on LibriSpeech train-clean-100/dev-clean and reports LibriSpeech test-clean/test-other, with and without a 4-gram LM. ROB-91 trains/evaluates TEDLIUM and reports no LM-assisted result.
- The ROB-91 output is dominated by deletions in the CSV, consistent with a near-blank CTC behavior. That could mean the probe failed to optimize, the frozen features are weak, the final-layer features are not linearly/locally useful, or the evaluation domain/setup is mismatched.

Interpretation: the final ROB-91 result is a real negative signal for the current checkpoint plus current probe/eval setup, but it should not be treated as a clean falsification of BEST-RQ-style SSL. The probe is good enough to justify investigating the SSL setup, especially masking, but not good enough to isolate the exact failure alone.

### 5. How Low Should BEST-RQ CE Go?

With an 8192-entry codebook, a uniform random classifier has CE `log(8192) ~= 9.01`. A drop to about `5.7` means the model is predicting random-projection targets far above chance. That is not inherently suspicious for BEST-RQ with random codebooks; the point of the method is that random fixed targets still create a useful masked prediction task.

However, CE is not directly comparable across different mask densities, input visibility, batch shapes, and target distributions. A low or decreasing CE under 10% masking can be easier than the paper's 48-60% actual-mask regime and may not imply downstream transfer. The current loss curve is therefore not sufficient evidence that the SSL run learned useful ASR representations.

## Ranked Findings

1. Most likely training problem: actual mask density is much lower than the open implementation. This is both code-backed and paper-backed, and the paper's ablation says masking ratio strongly affects downstream ASR. The precise comparison is 10% actual masking here versus about 48% actual for the best Table 3 ablation row and about 60% actual for the main SpeechBrain/open setting.
2. Likely training/setup problem: self-conditioning is active during SSL pretraining, so our model is not the same "encoder-only" setup as SpeechBrain BEST-RQ. This may be acceptable for LCASR, but it should be an explicit experiment variable.
3. Likely evidence problem: the ROB-70 checkpoint step labels are not optimizer-step labels. The run may be materially undertrained relative to the paper even if the checkpoint names look large.
4. Moderate probe limitation: ROB-91's final BiLSTM probe fixed the simple-head concern, but it still lacks hidden-state weighted sums and LM/no-LM LibriSpeech comparison.
5. Lower-confidence bug concern: no current evidence that the random-projection quantizer, target shape, or frozen checkpoint extraction is fundamentally broken.

## Recommended Next Work

Run these in order; do not start with another large pretraining run.

1. Add a mask-density diagnostic.
   - Log valid stacked frames, masked stacked frames, final mel-frame mask ratio, and skipped-empty-mask counts.
   - Run a tiny smoke and compare the observed ratio against the intended setting. The current config should be about 10% actual masking; paper-style diagnostics should distinguish 12% start / about 48% actual from 15% start / about 60% actual.

2. Patch and smoke a paper-style masking mode.
   - Implement start-index mask semantics: sample roughly `mask_prob * T` non-overlapping starts and mask `mask_length` following stacked frames.
   - First test `mask_prob: 0.12`, `mask_length: 4` because that matches the best Table 3 ablation row. Also test or keep available `mask_prob: 0.15`, `mask_length: 4` because that matches the final SpeechBrain/open recipe.

3. Run a short controlled SSL ablation before full rerun.
   - Current low-mask config versus paper-style mask.
   - Optionally self-conditioning on versus off.
   - Same data slice, same number of optimizer steps, same logging.
   - Compare CE, masked-frame counts, skip rates, and a very small frozen probe or linear separability diagnostic.

4. Make the probe closer to MP3S before final judgment.
   - Extract all hidden states or a selected layer set.
   - Train a learnable weighted sum plus 2-layer BiLSTM CTC head.
   - Prefer a LibriSpeech train-clean-100/dev-clean/test-clean/test-other setup if the goal is direct paper comparison; keep TEDLIUM as domain-transfer evidence.

5. Only then rerun full SSL.
   - Use the corrected mask setting and explicit self-conditioning decision.
   - Treat checkpoint labels as optimizer-step or sample-count labels explicitly in run manifests.
   - Keep W&B logs for mask density and skipped chunks, not just CE.

## Conclusion

The current evidence points first at SSL setup rather than the downstream probe alone. ROB-91 already repaired the most obvious probe issue by using the paper-style BiLSTM head for 10 epochs, and the result stayed near 100% WER. But the SSL objective differs from the open implementation in a way that is likely to matter: it masks about 10% actual stacked frames, while the paper's successful settings are roughly 48-60% actual masking depending on whether the comparison target is the best Table 3 ablation row or the final SpeechBrain/open recipe. That can explain a decreasing CE curve that does not transfer to CTC probing.

The next useful experiment is a small paper-style masking diagnostic/ablation, not a blind longer rerun of the same config.
