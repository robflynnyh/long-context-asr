# ROB-98 Poor SSL Performance Report

Date: 2026-05-20

Post-ROB-119 addendum: 2026-05-22

Post-ROB-125/128 addendum: 2026-05-22

Post-ROB-128 completion addendum: 2026-05-23

Post-ROB-126 completion addendum: 2026-05-24

Post-ROB-129 completion addendum: 2026-05-24

Post-ROB-129 old-mask comparison plan: 2026-05-24

Post-ROB-130 completion addendum: 2026-05-24

## Scope

This report investigates why the ROB-91 frozen probes over the ROB-70 BEST-RQ SSL checkpoints performed poorly. It compares the repository implementation against the open BEST-RQ implementation described in arXiv:2405.04296 and the current SpeechBrain BEST-RQ recipe.

Required preflight was completed before planning or editing:

- `symphony/instructions/linear-context.md`: required a recent Linear comment reread. The initial report was written when ROB-98 had no comments; later human clarifications asked where the 60% masking claim came from, then asked that TEDLIUM probe training use default utterance boundaries instead of repo chunking and that the small probe batch sizes be corrected.
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
- ROB-100 corrected SSL run evidence from `symphony/RESEARCH_DIARY.md`
- ROB-119 weighted BiLSTM probe summary: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/rob119-full-weighted-bilstm-20260522T110547Z/OUTCOME.md`
- ROB-119 weighted BiLSTM probe manifest and diagnostics:
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/rob119-full-weighted-bilstm-20260522T110547Z/run_manifest.json`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/rob119-full-weighted-bilstm-20260522T110547Z/diagnostics/primary.jsonl`
- ROB-119 implementation branch inspected at `origin/symphony/rob-119-paper-matched-bestrq-probe`
- ROB-125/ROB-128 Linear comments on the supervised-feature probe, unsorted TEDLIUM train timestamps, and the newer utterance-boundary correction.
- ROB-128 utterance-boundary overfit summary: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-128/rob128-utterance-ladder-normfix-20260523T161617Z/OUTCOME.md`
- ROB-128 stage-scoped ablation summary: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-128/rob128-ablation-stagefix-20260523T165221Z/ABLATION.md`
- ROB-126 interrupted top-layer-adaptation summary: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126/rob126-step24416-snapshot-eval-rootfix-20260524T0412Z/OUTCOME.md`
- ROB-126 reusable TEDLIUM utterance cache sentinel: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126/tedlium_train_utterances/_SUCCESS.clean_stm_target_v2.json`
- ROB-129 corrected fully frozen ROB-100 probe summary: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-129/rob129-full-frozen-b32-lr1e3-4epoch-wandb-ckptfix-20260524T110914Z/OUTCOME.md`
- ROB-129 corrected fully frozen ROB-100 probe CSV and diagnostics:
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-129/rob129-full-frozen-b32-lr1e3-4epoch-wandb-ckptfix-20260524T110914Z/tedlium_test_results.csv`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-129/rob129-full-frozen-b32-lr1e3-4epoch-wandb-ckptfix-20260524T110914Z/diagnostics/random_linear.jsonl`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-129/rob129-full-frozen-b32-lr1e3-4epoch-wandb-ckptfix-20260524T110914Z/diagnostics/random_bilstm.jsonl`
- ROB-91 old low-mask SSL source-checkpoint cache:
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/source-checkpoints/step_25344.pt`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/source-checkpoints/step_52800.pt`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/source-checkpoints/step_105360.pt`
- ROB-91 old chunked TEDLIUM probe summary:
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/rob91-full-bilstm-10epoch-constant-20260519T0911Z/OUTCOME.md`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/rob91-full-bilstm-10epoch-constant-20260519T0911Z/run_manifest.json`
- ROB-130 corrected old low-mask ROB-70 probe summary:
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-130/rob130-lowmask-100pct-b32-lr1e3-4epoch-20260524T140324Z/OUTCOME.md`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-130/rob130-lowmask-100pct-b32-lr1e3-4epoch-20260524T140324Z/tedlium_test_results.csv`
- `symphony/scripts/prepare_rob91_tedlium_manifest.py`
- `exp/train.py`
- `eval/tedlium/run.py`

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

## Post-ROB-119 Addendum

ROB-100 and ROB-119 tested the strongest recommendation from the initial report:

- ROB-100 repeated BEST-RQ SSL with paper-style masking: `mask_mode=speechbrain`, `mask_prob=0.12`, `mask_length=4`, about `0.48` actual masked stacked frames on full chunks, and self-conditioning disabled.
- ROB-100 completed one Spotify SSL epoch with `65523` logged loss updates. Its first-100 loss mean was `8.2474`, last-100 mean was `5.7960`, and final logged loss was `4.8904`.
- ROB-119 then probed `/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520/step_105360.pt` with a stronger frozen setup than ROB-91: trainable weighted sum over six exposed encoder-layer hidden states, a 2-layer BiLSTM CTC head, hidden size `1024`, dropout `0.2`, TEDLIUM training/eval, and explicit blank/deletion diagnostics.

The ROB-119 result was still poor:

| Probe | Checkpoint | Probe setup | Evidence type | Result |
| --- | --- | --- | --- | --- |
| ROB-91 | ROB-70 low-mask SSL | final hidden state, 2-layer BiLSTM, 10 epochs | TEDLIUM transfer | `99.73%` WER |
| ROB-119 | ROB-100 paper-mask SSL | weighted hidden-state sum, 2-layer BiLSTM, 10 epochs | TEDLIUM transfer | `99.61%` WER, `92.44%` deletions, `0.00%` insertions |

ROB-119 is therefore a real negative follow-up, but it changes the diagnosis rather than simply invalidating the original report. The earlier low-mask finding was still a genuine bug/mismatch: ROB-100 fixed it and the SSL loss learned the paper-style random-code objective. The new evidence says that fixing masking alone did not make this one-epoch Spotify checkpoint useful under a frozen TEDLIUM CTC transfer probe.

### What ROB-119 Rules Out

ROB-119 makes these explanations less likely as sole causes:

1. **"ROB-91 failed only because it used the final hidden state."** ROB-119 added a learnable weighted hidden-state sum and improved WER by only `0.12` absolute points.
2. **"ROB-91 failed only because the probe head was too weak."** Both the final ROB-91 run and ROB-119 used the 2-layer BiLSTM head that matches the open BEST-RQ / MP3S ASR probe description.
3. **"The corrected masking setting alone was sufficient."** ROB-100 hit the intended high actual mask ratio and learned the SSL CE objective, but the downstream frozen probe remained deletion-dominated.

### What ROB-119 Does Not Rule Out

ROB-119 still leaves several important possibilities open:

1. **Probe optimization is still not fully settled.** ROB-119 training diagnostics show learning rather than a dead run: epoch mean loss fell from `14.3364` to `10.7474`, and mean blank probability moved from `0.9607` to `0.9366`. The canceled LR-grid attempt was stopped by request before it could produce a clean answer. However, the completed 10-epoch evaluation stayed near-empty, so probe optimization alone would need to explain a very large deletion collapse.
2. **TEDLIUM transfer is not direct paper evidence.** The open implementation reports LibriSpeech train-clean-100/dev-clean probing and test-clean/test-other evaluation, with and without a 4-gram LM. ROB-119 is explicitly TEDLIUM transfer evidence. Domain mismatch and no LM can hurt, but they do not by themselves explain hypotheses with only `2133` words against `28215` reference words.
3. **The frozen SSL representation may be weak even if the SSL loss decreases.** BEST-RQ CE can improve by learning random-code prediction from local acoustic context without necessarily exposing CTC-usable phonetic features to a frozen probe, especially with different data, architecture, optimizer, chunking, and update horizon from the open recipe.
4. **The original BEST-RQ paper is not a frozen-probe setup.** Chiu et al. report strong ASR transfer from random-projection targets, but in a downstream ASR training/fine-tuning context. The frozen weighted-state MP3S probe in arXiv:2405.04296 is the closest public comparison for ROB-119, not the original paper's full ASR fine-tuning recipe.

### Revised Hypothesis Ranking

1. **Most likely: frozen representation quality is the blocker under this local SSL setup.** The corrected paper-mask checkpoint learns the SSL objective, but a stronger frozen weighted-state BiLSTM probe still emits very few words. This points to representations that are not directly CTC-usable after one local SSL epoch, not just to the old low-mask bug.
2. **Likely: the SSL recipe/horizon is still not close enough to the successful open setup.** The open implementation pretrains on LibriSpeech 960h for 42 epochs or roughly 200k steps, uses a 12-layer conformer, AdamW with Noam warmup, dynamic batching, and evaluates both 100k and 200k checkpoints. ROB-100 uses one Spotify epoch, 6 SCConformerXL layers, the local optimizer/scheduler path, fixed chunking, and about 65k logged updates.
3. **Possible: the TEDLIUM probe needs a different optimization recipe, but this is now a secondary hypothesis.** The loss was still improving at epoch 10, so a longer or LR-tuned probe could improve. But the completed run's `99.61%` WER and deletion-heavy output mean this is unlikely to be a small scheduler-only issue.
4. **Possible: the probe harness itself has a hidden issue.** This is lower-confidence because ROB-119 did load/freeze the checkpoint, expose six hidden states, train a weighted sum and BiLSTM head, and produce diagnostics. Still, a known-good frozen supervised encoder sanity probe has not yet demonstrated that the exact ROB-119 TEDLIUM CTC path can produce non-blank output when the features are known to be useful.
5. **Less likely: random-codebook BEST-RQ is intrinsically unsuitable.** Both the original BEST-RQ paper and the open implementation show useful ASR performance with fixed random projection/codebook targets. The local negative result is better explained by recipe, horizon, data, architecture, or probe/eval mismatch.

### Next Discriminating Checks

Do not start with another blind full SSL rerun. The next checks should separate probe viability from frozen SSL representation quality:

1. **Known-good frozen encoder sanity probe.**
   - Plain-English meaning: take an ASR checkpoint that is already known to contain useful supervised ASR features, freeze its encoder, and run the same ROB-119 CTC probe/eval path on top of it.
   - Use the exact ROB-119 TEDLIUM CTC path with a supervised checkpoint already known to decode or fine-tune well in this repo.
   - Freeze the encoder and train the same weighted-state/BiLSTM probe where possible.
   - If this also stays blank, debug the probe harness, CTC labels, LR, batching, or eval path before drawing more SSL conclusions.
   - If it emits words, the ROB-119 path is viable and the ROB-100 frozen SSL representation is the likely blocker.

2. **Small top-N-unfrozen ROB-100 probe.**
   - Plain-English meaning: start from ROB-100 again, but do not freeze the whole SSL model. Unfreeze only the last one or two encoder layers and train those layers together with the weighted-state/BiLSTM CTC head.
   - Start from the ROB-119 config but unfreeze only the top one or two encoder layers, optionally with a lower encoder LR than the BiLSTM/weighted-sum LR.
   - If this quickly escapes deletion collapse, the SSL checkpoint contains some useful low/mid-level information but its frozen final representation is not linearly/recurrently accessible enough.
   - If it remains blank, the issue is probably deeper: data/labels/eval, insufficient SSL pretraining, or architecture/objective mismatch.

3. **Short LibriSpeech direct-comparison probe if data paths are available.**
   - This would remove the TEDLIUM transfer caveat and compare against the open implementation's ASR probe target.
   - If LibriSpeech is blocked locally, keep all future probe results labeled as transfer evidence rather than paper-matched ASR evidence.

4. **Only after those checks, consider a longer or more paper-like SSL repeat.**
   - A useful repeat should target update count and recipe match, not just raw audio hours. Concretely: more optimizer updates, AdamW/Noam-style settings, explicit checkpoint labels by optimizer update, and possibly a 12-layer architecture if resources allow.
   - Continue logging actual mask ratio, skipped-empty-mask count, SSL CE, and downstream probe diagnostics.

## Post-ROB-125/128 Addendum: TEDLIUM Probe Training Unit Is Likely Wrong

The latest ROB-98 human comment changes the interpretation again. ROB-128 found a real bug in the chunk-label path: ROB-125 TEDLIUM training transcript JSONs were mostly out of timestamp order, and `chunk_text_json()` assumes chronological entries. Sorting would improve chunk labels, but the newer instruction is stronger: for TEDLIUM supervised probe training, use the default TEDLIUM utterance boundaries rather than this repo's long-recording chunking setup.

That is consistent with the code:

- `symphony/scripts/prepare_rob91_tedlium_manifest.py` builds one manifest item per TEDLIUM `.sph` recording and writes all STM segments into `word_timestamps`.
- `exp/train.py` loads that manifest through `VariableBatchSimpleDataloader`, slices each whole recording into fixed `audio_chunking.size` chunks, and calls `chunk_text_json()` to create a CTC target for each chunk.
- `eval/tedlium/run.py` already knows the default TEDLIUM utterance unit through `fetch_utterances(...)`: it slices each STM segment by start/end frame and keeps the original segment text.

So the ROB-91/119/125 probe training path was not using the same natural training unit as TEDLIUM evaluation. It was training a fresh CTC head on artificial fixed-width slices of full talks, with labels reconstructed by time-window inclusion. That is fragile even after sorting: utterances crossing chunk boundaries can disappear or be split poorly, silent chunks are skipped, and the model sees an optimization problem that differs from ordinary utterance-level ASR probe training.

This also explains the otherwise odd ROB-125 split:

| Evidence | What it says |
| --- | --- |
| ROB-125 source supervised CTC smoke: `8.06%` WER on one TEDLIUM record through the moving-window/greedy path | The checkpoint and eval path can emit normal words. |
| ROB-125 fresh weighted-BiLSTM probe on the same known-good supervised features: `99.53%` WER, `89.87%` deletions | Fresh probe training is still broken or badly mismatched. |
| ROB-128 timestamp audit | The current chunk-label construction was definitely corrupt for most records before sorting. |
| Latest human boundary correction | Sorting is not the full fix; TEDLIUM probe training should avoid chunk-label construction and use utterance segments directly. |

### Batch Size Correction

The small probe batch sizes are now another likely setup issue. ROB-91's wrapper defaulted the full probe batch to `16`, ROB-126 was launched with `ROB126_BATCH_SIZE=8`, and ROB-128 queued a one-record overfit with `ROB128_BATCH_SIZE=1`. Those values are much smaller than the open BEST-RQ setting's dynamic batching regime. The paper says the open experiments used dynamic batches of about `100` seconds per GPU across 8 V100s, and the current SpeechBrain recipe uses dynamic length batching rather than a tiny fixed example count.

The direct fix is not simply "set batch size to one huge number" for full talks. Once TEDLIUM is converted to utterance-level samples, the average sequence is much shorter and the batch can be increased substantially. A practical next run should either:

- use dynamic/bucketed batching by utterance duration; or
- start with a conservative fixed utterance batch such as `64` or `128`, then increase until GPU memory is close to full.

This matters because CTC probe training with very small batches can get a biased blank-heavy gradient, especially with a randomly initialized BiLSTM/CTC head and no LM.

### Child Runs Corrected

Because the latest instruction made the active child setup stale, I stopped the obsolete child runs instead of letting them consume GPU time:

- ROB-126 `rob126-full-top2-4epoch-20260522T171921Z`: stopped after it had launched on Mimas GPU 2. It used `ROB126_BATCH_SIZE=8` and the chunked TEDLIUM manifest path.
- ROB-128 `rob128-overfit-bilstm-sortfix-20260522T211416Z`: removed while still waiting in the `with-gpu` queue. It used the one-record/chunked-label overfit design with `ROB128_BATCH_SIZE=1`.

Both child issues were moved back to `Todo` with comments instructing the next worker to rebuild around TEDLIUM utterance-level training examples before rerunning.

### Revised Current Diagnosis

The parent investigation now has two concrete probe-training blockers, not just a vague "fresh CTC probe may be bad" hypothesis:

1. **TEDLIUM probe labels were constructed with the wrong training unit.** The repo treated full TEDLIUM talks like long-context training recordings and generated fixed-window CTC labels. TEDLIUM probe training should instead use STM utterance boundaries as the sample boundaries.
2. **The existing chunk-label path also had a timestamp-order bug.** ROB-128 showed that sorting changes labels for almost all ROB-125 training records. This is real, but it is now secondary because the corrected design should avoid chunking for TEDLIUM probe training.
3. **Probe batches were too small.** The completed and queued probes used fixed example counts between 1 and 16. Once utterance-level samples are used, increase batch size materially or switch to duration-based dynamic batching.
4. **SSL quality is still unresolved.** ROB-100 fixed the SSL masking mismatch and learned the SSL objective, but the downstream evidence is confounded until the supervised-feature probe works with clean utterance-level TEDLIUM training.

### Replacement Plan

The next ROB-126/128 work should use this order:

1. Build a TEDLIUM utterance-level training manifest.
   - One sample per non-ignored STM segment.
   - Audio is `processing_chain(sph_path)[:, :, start_frame:end_frame]` or a cached equivalent.
   - Text is the exact normalized STM segment text.
   - No `chunk_text_json()` and no fixed 2048-frame training chunks for this supervised probe.

2. Add a small loader smoke.
   - Print/record the first few IDs, durations, token lengths, and decoded labels.
   - Assert there are no zero-token training examples except deliberately filtered silence/ignored segments.
   - Confirm padding/batching uses utterance lengths, not full-talk chunk lengths.

3. Run the known-good supervised-feature probe first.
   - Use the ROB-125 supervised checkpoint.
   - Freeze the encoder/backbone and train the same weighted-state BiLSTM CTC head.
   - Start with a much larger utterance batch than `8` if memory allows.
   - The acceptance criterion is not final WER; it is escaping near-total deletion collapse and showing that the exact fresh-probe path can learn from known-good features.

4. Only after that, rerun ROB-126.
   - Reuse the utterance-level path.
   - Start from ROB-100, unfreeze top encoder layers as before, and keep the GPU 1/2 constraint if still required by current human comments.

5. Reinterpret ROB-119/ROB-125 frozen-probe WER as stale until this is done.
   - The source supervised CTC eval evidence remains valid.
   - The fresh-probe results are not clean evidence against ROB-100 SSL representation quality until the TEDLIUM training unit and batch-size issues are fixed.

### Updated Conclusion

The initial ROB-98 conclusion should be revised, not discarded. Low actual mask density was the highest-confidence mismatch in ROB-70 and it was worth fixing. ROB-100 fixed that mismatch, but ROB-119 shows that the corrected one-epoch checkpoint still does not yield useful frozen TEDLIUM CTC representations under a substantially more paper-matched probe.

The most actionable next move is now narrower: fix the supervised TEDLIUM probe training path before interpreting any more frozen SSL probe WER. Use default TEDLIUM utterance boundaries, increase the practical utterance batch size, and first prove that the fresh weighted-state BiLSTM CTC probe can learn from a known-good supervised encoder. Only then should ROB-126's top-layer-unfrozen ROB-100 probe or a longer paper-like SSL rerun be used as evidence about SSL representation quality.

## Post-ROB-128 Completion Addendum: Probe Path Is Viable, But Head Optimization Is Sensitive

ROB-128 has now run the corrected supervised-control ladder on known-good ROB-81 supervised features with TEDLIUM train examples cut on STM utterance boundaries and normalized before tokenization. This resolves one parent-level question: the corrected utterance-boundary probe path can train a fresh CTC head to emit real words when the frozen encoder is known to be useful.

Key ROB-128 results:

| Evidence | Result | Parent-level interpretation |
| --- | --- | --- |
| One-record normalized-label random-BiLSTM overfit, 80 epochs | `0.23%` WER, `0.31%` CER, `49.46%` blank, `2655` hyp words / `2649` ref words | The fresh probe stack can learn from known-good supervised features once TEDLIUM labels and utterance boundaries are correct. This rules out a universal decode/CTC-label-path failure. |
| Source CTC eval-only on the same one-record utterance set | `22.69%` WER, `81.57%` blank, `2247` hyp words / `2649` ref words | The supervised checkpoint is useful under the utterance-level eval path, though its source projection is not tuned for this exact one-record normalized probe setup. |
| Stage-scoped 20-epoch source-linear trainable head | `10.49%` WER | Starting from a useful supervised CTC projection adapts cleanly. |
| Stage-scoped 20-epoch random-linear head | `43.07%` WER | A simple fresh head can learn usable output under the corrected labels, although not as well as the source-initialized head. |
| Stage-scoped 20-epoch fresh random-BiLSTM head | `100.00%` WER, `100.00%` blank, `0` hyp words | The BiLSTM probe recipe is still highly sensitive to initialization/training horizon/optimization. The earlier blank-heavy BiLSTM failures cannot be read as pure representation-quality evidence. |

The important correction is that ROB-128 is not "probe fixed, SSL bad." It is more precise:

1. **TEDLIUM probe data/label construction was a real blocker and is now corrected in ROB-128.** The one-record overfit result would not be possible if the utterance-boundary loader, normalized labels, CTC tokenization, or greedy eval path were fundamentally broken.
2. **The known-good supervised-control result still points to probe-head sensitivity.** A source-initialized linear head and a random linear head both learn on the corrected one-record setup, while the fresh random BiLSTM collapses at 20 epochs and only proves overfit success after the longer gated 80-epoch run.
3. **ROB-119/ROB-125 frozen SSL WER remains stale evidence.** Those runs used the old full-recording/chunk-label TEDLIUM setup and small fixed batches, so they should not be treated as final evidence against ROB-100 SSL representation quality.
4. **ROB-100 SSL quality remains unresolved.** The parent issue has now isolated a corrected probe path, but that path still needs to be applied to ROB-100/ROB-126 with the same diagnostics before drawing a conclusion about the BEST-RQ checkpoint.

### Current Recommendation

Use the ROB-128 corrected utterance-boundary infrastructure as the gate for any further ROB-100 interpretation:

1. Start the next ROB-100/ROB-126 probe with a conservative head that already proved learnable on supervised features, preferably source-style linear initialization where possible and a random-linear control.
2. Keep the fresh BiLSTM probe as a separate stress test, not the sole acceptance criterion. If used, give it enough horizon and log blank probability, hypothesis word count, and train loss at each stage.
3. Keep the result labeled as TEDLIUM transfer evidence unless a LibriSpeech train-clean-100/dev-clean/test-clean/test-other path is added for direct comparison to the open BEST-RQ/MP3S setup.
4. Do not launch another large SSL rerun from the parent issue until the corrected probe ladder has been run against ROB-100. The original SSL masking diagnosis was real, but current uncertainty is dominated by probe setup and head optimization rather than a known remaining SSL-code bug.

## Post-ROB-126 Completion Addendum: Top-Layer Adaptation Escapes Deletion Collapse

ROB-126 has now applied the corrected TEDLIUM utterance-boundary setup to ROB-100 with the top two encoder layers unfrozen. The final handoff is not a full four-epoch completion: the human stopped the run after it was clearly emitting words, and the evaluated checkpoint is an interrupted snapshot at epoch `0`, step `24416`. Even with that caveat, it is a useful discriminator.

Key ROB-126 evidence:

| Evidence | Result | Parent-level interpretation |
| --- | --- | --- |
| ROB-126 top-2-unfrozen ROB-100 snapshot | `65.37%` WER, `46.60%` CER, `1.95%` insertions, `21.83%` deletions, `41.59%` substitutions | Small supervised adaptation of the SSL encoder can emit substantial nonblank output under the corrected TEDLIUM utterance setup. |
| ROB-119 frozen weighted BiLSTM baseline | `99.61%` WER, `92.44%` deletions | The old frozen-probe result remains a deletion-collapse baseline, but it used the stale chunked TEDLIUM path. |
| Difference versus ROB-119 | `-34.24` WER points and `-70.61` deletion-rate points | The ROB-100 checkpoint is not obviously useless once the corrected probe data path and limited encoder adaptation are allowed. |
| ROB-126 reusable utterance cache | `56,803` train utterances from `774` TEDLIUM recordings, `clean_stm_target_v2`, `clone_contiguous_slice` | The next frozen-probe rerun should reuse this cache instead of rebuilding full talks or saving slice views. |

This result changes the next action but does not close the parent diagnosis. It strengthens the case that ROB-100 learned something useful, because top-layer adaptation quickly moved away from near-total deletion collapse. It does not yet prove that the fully frozen ROB-100 representation is CTC-usable, because the successful ROB-126 path trained layers `4` and `5` of the SSL encoder plus the probe head.

The newest ROB-98 human comment asks to test the normal frozen probe again using the corrected TEDLIUM utterances. That has been split into ROB-129: https://linear.app/robflynn/issue/ROB-129/rerun-frozen-rob-100-probe-with-corrected-tedlium-utterances

ROB-129 should be treated as the next parent-level gate:

1. Reuse `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126/tedlium_train_utterances` if the `_SUCCESS.clean_stm_target_v2.json` sentinel and loader smoke still pass.
2. Keep ROB-100 frozen; do not unfreeze top layers in ROB-129.
3. Start with the corrected ROB-126/ROB-128 utterance-folder training path, not the old ROB-119 full-recording/chunked-label path.
4. Include a learnable head/control that ROB-128 showed can optimize under corrected labels where possible, and keep the fresh BiLSTM as a stress test rather than the only acceptance criterion.
5. Report blank probability, hypothesis/reference word counts, deletion rate, train loss, WER/CER, and exactly which hidden states or middle-layer states were exposed to the probe.

### Current Parent Interpretation

The investigation is now less consistent with a single "SSL setup is broken" explanation. The original ROB-70 low-mask bug was real, ROB-100 fixed it, ROB-128 fixed the TEDLIUM probe training unit, and ROB-126 shows that limited encoder adaptation can extract useful output from ROB-100. The remaining unresolved question is narrower: whether ROB-100's frozen hidden states are useful under the corrected TEDLIUM utterance-level probe, and if so which head/layer selection exposes them. ROB-129 is the direct test of that question.

## Post-ROB-129 Completion Addendum: Corrected Frozen ROB-100 Probe Works, But Is Still Weak

ROB-129 has now run the direct parent-level gate requested after ROB-126: keep ROB-100 fully frozen and rerun the TEDLIUM probe on the corrected STM utterance-boundary training cache. This makes the old ROB-119 frozen-probe result stale as evidence against ROB-100. The corrected frozen setup no longer deletion-collapses.

Key ROB-129 setup:

- SSL checkpoint: ROB-100 paper-mask checkpoint `/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520/step_105360.pt`, locally cached at `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126/source-checkpoints/step_105360.pt`.
- Frozen state: `unfreeze_top_n_layers: 0`; the encoder/backbone stayed frozen.
- Trainable probe: weighted sum over six post-layer SCConformerXL hidden states from layers `0-5`, plus either a random linear CTC head or random 2-layer BiLSTM CTC head.
- Training data: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126/tedlium_train_utterances`, validated by `_SUCCESS.clean_stm_target_v2.json` with `56,803` utterance files.
- Optimization: four epochs, batch size `32`, constant LR `1e-3`, no warmup, final-only practical checkpoint retention.

Key ROB-129 results:

| Probe | Frozen encoder? | Head | WER | CER | Insertions | Deletions | Substitutions | Hyp/ref words | Final loss | Final blank p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROB-129 random linear | yes | weighted states + linear CTC | `67.17%` | `48.75%` | `1.85%` | `27.94%` | `37.38%` | `20,855 / 28,215` | `15.7506` | `90.02%` |
| ROB-129 random BiLSTM | yes | weighted states + 2-layer BiLSTM CTC | `38.14%` | `22.42%` | `3.32%` | `9.84%` | `24.97%` | `26,375 / 28,215` | `7.9920` | `82.63%` |
| ROB-126 top-2-unfrozen snapshot | no, layers `4-5` trainable | weighted states + probe | `65.37%` | `46.60%` | `1.95%` | `21.83%` | `41.59%` | not the same final frozen comparison | not final | not final |
| ROB-119 stale frozen baseline | yes | old chunked-label weighted BiLSTM | `99.61%` | not recorded here | `0.00%` | `92.44%` | about `7.17%` | `2,133 / 28,215` | stale | stale |

Interpretation:

1. **The corrected frozen ROB-100 representation is CTC-usable under TEDLIUM transfer.** The best ROB-129 row improves by `61.47` WER points and `82.60` deletion-rate points versus the stale ROB-119 frozen result. The hypotheses contain `26,375` words against `28,215` reference words, so this is no longer a blank/deletion-collapse failure.
2. **The old probe-training data path, not just SSL quality, caused most of the dramatic failure.** ROB-119 used the full-recording/chunked TEDLIUM path; ROB-129 used corrected STM utterance examples. That change turns a near-empty frozen probe into a readable, measurable ASR system.
3. **Fully frozen ROB-100 is still not strong by ASR standards.** `38.14%` TEDLIUM test WER is much worse than a supervised TEDLIUM model and still far from the open implementation's LibriSpeech numbers. This should be read as "the representation has useful signal" rather than "the SSL setup is now good enough."
4. **The BiLSTM head is useful once the data path is fixed.** ROB-128 showed fresh BiLSTM optimization can be fragile; ROB-129 shows that, with four epochs and batch `32` on the corrected cache, the BiLSTM head is much better than the random linear head (`38.14%` versus `67.17%` WER).
5. **ROB-126 and ROB-129 now tell a coherent story.** Limited top-layer adaptation helps, but the fully frozen model also contains usable information. The parent question has moved from "why does everything collapse?" to "why is the corrected frozen transfer number still weak?"

### Current Diagnosis After ROB-129

The investigation should no longer rank "the SSL checkpoint is useless" or "the frozen probe cannot work" as primary explanations. The corrected evidence supports this narrower diagnosis:

1. **Confirmed original SSL mismatch:** ROB-70 used much lower actual masking than the open implementation; ROB-100 fixed this with paper-style masking and remains the correct SSL checkpoint for follow-up.
2. **Confirmed probe data bug:** TEDLIUM probe training on full-recording chunks was the biggest cause of the earlier near-100% WER/deletion collapse. Correct STM utterance-boundary training is mandatory for these probes.
3. **Current residual SSL/recipe gap:** Even with corrected masking and corrected frozen probing, ROB-100 reaches only `38.14%` WER on TEDLIUM transfer. Remaining differences from the open implementation are still substantial: Spotify rather than LibriSpeech pretraining, one local epoch rather than roughly 100k-200k optimizer steps, 6-layer SCConformerXL rather than a 12-layer SpeechBrain conformer, different optimizer/scheduler/batching, and TEDLIUM transfer rather than LibriSpeech train-clean-100 probing.
4. **Current residual probe gap:** The best corrected probe is still TEDLIUM transfer with no LM, not the paper's LibriSpeech train-clean-100/dev-clean/test-clean/test-other setup with optional 4-gram LM. It is useful project evidence, but not a direct paper reproduction.

### Recommended Next Checks

The next checks should be targeted; another blind rerun is not justified yet.

1. **Make ROB-129 the new baseline.**
   - Treat `38.14%` WER / `22.42%` CER / `9.84%` deletions as the corrected fully frozen ROB-100 TEDLIUM-transfer baseline.
   - Retire ROB-119/ROB-91 frozen WER as stale evidence except when explicitly discussing the old broken chunk-label setup.

2. **Run a matched supervised-reference probe on the full corrected TEDLIUM train/test setup.**
   - ROB-128 proved one-record overfit and stage-scoped controls, but the parent now needs a full TEDLIUM corrected-cache reference for a known-good supervised encoder using the same four-epoch, batch-32, weighted-state probe recipe.
   - This gives a scale for "38.14% WER": if a known-good frozen supervised encoder gets much lower WER under the same recipe, ROB-100 is still representation-limited; if it is also high, the probe/eval recipe still needs tuning.

3. **Compare frozen layer/weight behavior.**
   - Inspect learned hidden-state weights from the ROB-129 BiLSTM probe.
   - Run cheap layer-selection probes if needed: final layer only, middle layer only, and learned weighted sum. This can show whether the useful signal is concentrated in lower/middle layers.

4. **Only then choose between longer SSL and recipe-matched SSL.**
   - If the supervised-reference probe is strong and ROB-129 remains at `38.14%`, prioritize SSL recipe/horizon work: longer ROB-100-style training, optimizer/scheduler parity, or more LibriSpeech-like data.
   - If the supervised-reference probe is also weak, prioritize probe recipe changes: duration batching, LR/horizon search, source-initialized controls, and LM/no-LM eval comparison.

5. **Keep paper comparison labels precise.**
   - ROB-129 is corrected TEDLIUM transfer evidence.
   - A direct arXiv:2405.04296 comparison still requires LibriSpeech train-clean-100/dev-clean probing and test-clean/test-other evaluation, ideally with the no-LM and 4-gram LM split used by the paper.

### Updated Conclusion

The parent issue has identified two real, high-impact causes: the original SSL masking mismatch and the TEDLIUM probe training-unit bug. After both were corrected, the fully frozen ROB-100 checkpoint is no longer a collapse case: the corrected weighted-state BiLSTM probe reaches `38.14%` WER and emits nearly the right number of words. That is the strongest evidence so far that ROB-100 learned usable speech structure.

The remaining poor-performance question is now about quality and comparability, not total failure. ROB-100 is still weak relative to supervised ASR and not directly comparable to the open BEST-RQ paper because the pretraining data, training horizon, model recipe, downstream dataset, and LM setting differ. The most useful next discriminator is a full corrected-cache known-good supervised reference probe, followed by layer-selection analysis and then a deliberate decision between longer/more paper-like SSL training and further probe optimization.

## Post-ROB-129 Old-Mask Comparison Plan: Reprobe ROB-70 Before Attributing The Gain To Masking

The newest human comment asks for a comparison against the old lower-masking setup. That is the right next discriminator. The current report should not imply that ROB-100's paper-style masking caused the whole improvement from ROB-119/ROB-91 to ROB-129, because two major variables changed:

1. ROB-100 changed the SSL setup from legacy low actual masking to paper-style masking.
2. ROB-129 changed the TEDLIUM probe setup from full-recording chunk labels to corrected STM utterance-boundary examples.

The old ROB-91 low-mask probe is therefore stale for the same reason ROB-119 is stale: it used the broken/chunked TEDLIUM training path. Its near-100% WER rows cannot answer whether low-mask ROB-70 would still fail once trained with the corrected ROB-126 utterance cache and ROB-129-style frozen probe recipe.

### Exact Comparison Needed

Run a corrected frozen probe on the old ROB-70 low-mask checkpoints, using the same corrected TEDLIUM probe setup as ROB-129:

| Variable | ROB-129 current baseline | Needed old-mask comparison |
| --- | --- | --- |
| SSL checkpoint | ROB-100 paper-mask checkpoint `step_105360.pt` | ROB-70 low-mask checkpoint `step_105360.pt`, optionally also `step_25344.pt` and `step_52800.pt` |
| Masking during SSL | `mask_mode=speechbrain`, `mask_prob=0.12`, `mask_length=4`, about `48%` actual stacked-frame masking | legacy grouped `mask_percentage=0.1`, `frames_to_mask=5`, about `10%` actual stacked-frame masking |
| SSL self-conditioning | disabled in ROB-100 | enabled in the old ROB-70 probe configs/checkpoint family |
| Probe training data | corrected STM utterance cache `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126/tedlium_train_utterances` | same corrected STM utterance cache |
| Frozen state | fully frozen encoder/backbone, `unfreeze_top_n_layers: 0` | same fully frozen state |
| Probe head | weighted six-layer hidden-state sum plus random linear and random BiLSTM heads | same heads, same hidden-state exposure, same diagnostics |
| Current result | best row `38.14%` WER / `22.42%` CER / `9.84%` deletions | not yet measured under corrected TEDLIUM |

The minimum useful run is the old 100% ROB-70 checkpoint:

- Source checkpoint: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/source-checkpoints/step_105360.pt`
- Original external path recorded by ROB-91: `/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/6l_2048_1epoch_lr3e4_20260516/step_105360.pt`
- Stale old result under chunked TEDLIUM: `99.73%` WER from ROB-91.

If time allows, include the old 25% and 50% checkpoints from the same cache:

- `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/source-checkpoints/step_25344.pt`
- `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/source-checkpoints/step_52800.pt`

### How To Interpret The Result

This comparison cleanly separates the two confirmed fixes:

- If corrected low-mask ROB-70 remains far worse than ROB-129, then the original masking diagnosis is still a major quality explanation after controlling for the TEDLIUM probe bug.
- If corrected low-mask ROB-70 is close to ROB-129, then the dramatic old failure was mostly the TEDLIUM probe path, and ROB-100's paper-style masking may have helped less than expected under the current one-epoch Spotify recipe.
- If corrected low-mask ROB-70 beats ROB-129, inspect confounds before concluding low masking is better: ROB-70 used self-conditioning, different checkpoint history, and the old final decoder/probe loading path differs from the ROB-100 self-conditioning-off setup.

The comparison should be labeled as TEDLIUM transfer evidence, not a direct arXiv:2405.04296 reproduction. It still does not remove differences in pretraining data, update horizon, model architecture, optimizer/scheduler, and LM/no-LM evaluation.

### Updated Next Action

This has been split into ROB-130: https://linear.app/robflynn/issue/ROB-130/probe-old-low-mask-best-rq-checkpoint-with-corrected-tedlium

ROB-130 should run the corrected old-mask probe rather than launching it from ROB-98. It should reuse the ROB-126 TEDLIUM utterance cache, pin the old ROB-70 source checkpoints from the ROB-91 cache, preserve ROB-129 diagnostics and final-only/pruned checkpoint retention, and report back to ROB-98 with the old-mask versus paper-mask table.

## Post-ROB-130 Completion Addendum: Old Low-Mask Checkpoint Is Only Modestly Worse Under Corrected Probing

ROB-130 has now run the discriminator requested after ROB-129: take the old ROB-70 low-mask checkpoint family that originally looked collapsed under ROB-91, keep the encoder fully frozen, and probe it through the same corrected TEDLIUM utterance-boundary cache and ROB-129-style weighted hidden-state recipe. The minimum useful comparison, the old `step_105360.pt` checkpoint, completed.

Key ROB-130 setup:

- SSL checkpoint: ROB-70 low-mask checkpoint `/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/6l_2048_1epoch_lr3e4_20260516/step_105360.pt`, locally cached at `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-91/source-checkpoints/step_105360.pt`.
- SSL masking: legacy grouped `mask_percentage=0.1`, `frames_to_mask=5`, about `10%` actual stacked-frame masking.
- Frozen state: `unfreeze_top_n_layers: 0`; no trainable encoder layers.
- Trainable probe: weighted sum over six post-layer SCConformerXL hidden states from layers `0-5`, plus either a random linear CTC head or random 2-layer BiLSTM CTC head.
- Training data: the corrected ROB-126 TEDLIUM utterance cache at `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126/tedlium_train_utterances`, validated by `_SUCCESS.clean_stm_target_v2.json` with `56,803` utterance files.
- Optimization: four epochs, batch size `32`, constant LR `1e-3`, final-only/pruned checkpoint retention.

Key corrected old-mask results:

| Probe | SSL checkpoint | Head | WER | CER | Insertions | Deletions | Substitutions | Hyp/ref words | Final loss | Final blank p |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROB-130 random linear | ROB-70 low-mask `step_105360.pt` | weighted states + linear CTC | `70.62%` | `52.98%` | `0.79%` | `35.47%` | `34.36%` | `18,429 / 28,215` | `15.8930` | `89.02%` |
| ROB-130 random BiLSTM | ROB-70 low-mask `step_105360.pt` | weighted states + 2-layer BiLSTM CTC | `40.86%` | `24.32%` | `3.29%` | `10.95%` | `26.62%` | `26,055 / 28,215` | `8.5135` | `83.83%` |

Direct comparison to the corrected ROB-129 paper-mask baseline:

| Head | ROB-130 low-mask | ROB-129 paper-mask | Difference |
| --- | ---: | ---: | ---: |
| Random linear WER | `70.62%` | `67.17%` | ROB-130 worse by `+3.45` absolute points |
| Random linear deletions | `35.47%` | `27.94%` | ROB-130 worse by `+7.53` absolute points |
| Random BiLSTM WER | `40.86%` | `38.14%` | ROB-130 worse by `+2.72` absolute points |
| Random BiLSTM deletions | `10.95%` | `9.84%` | ROB-130 worse by `+1.11` absolute points |

Interpretation:

1. **The old ROB-91 deletion collapse was mostly the stale TEDLIUM probe path.** Under corrected utterance-boundary training, the same old low-mask checkpoint no longer behaves like a blank system. The BiLSTM probe emits `26,055` words against `28,215` reference words and reaches `40.86%` WER instead of the old near-`100%` WER.
2. **Paper-style masking still looks better, but only modestly in this corrected comparison.** ROB-129 beats ROB-130 by `2.72` WER points and `1.11` deletion points for the stronger BiLSTM head. That supports the original masking fix as a quality improvement, but it no longer explains the dramatic failure alone.
3. **The remaining quality gap is not specifically an old-mask collapse.** Both corrected frozen SSL probes are weak but usable: ROB-129 paper-mask at `38.14%` WER and ROB-130 low-mask at `40.86%` WER. The parent issue should now focus on why one-epoch local BEST-RQ transfer remains weak overall, not on why the old probes emitted almost no words.
4. **The comparison is still not a direct arXiv:2405.04296 reproduction.** It is TEDLIUM transfer evidence with no LM, Spotify pretraining, a 6-layer SCConformerXL backbone, one local epoch, and this repo's optimizer/scheduler/checkpoint history.

### Final Current Diagnosis

The investigation now has a clearer split between confirmed bugs and remaining research uncertainty:

1. **Confirmed and fixed SSL mismatch:** ROB-70's low actual mask density differed substantially from the open implementation. ROB-100's paper-style `mask_prob=0.12`, `mask_length=4`, self-conditioning-off run is still the cleaner recipe for future SSL work.
2. **Confirmed and fixed probe-data bug:** TEDLIUM probe training on full-recording chunks was the dominant cause of the near-total deletion collapse in ROB-91/ROB-119. Future TEDLIUM probes should use corrected STM utterance-boundary examples, normalized labels, larger practical utterance batches, and the ROB-126 cache contract when applicable.
3. **Refined masking interpretation:** Corrected low-mask probing is only slightly worse than corrected paper-mask probing for the strongest BiLSTM head. Masking matters, but the old result overstated its effect because the probe data path was broken.
4. **Remaining quality question:** Both corrected frozen SSL probes are still much weaker than a supervised ASR system and not directly comparable to the SpeechBrain/MP3S LibriSpeech setup. Plausible remaining causes are training horizon, data mismatch, architecture differences, optimizer/scheduler differences, no-LM TEDLIUM transfer, and probe optimization/layer-selection details.

### Final Recommended Next Checks

1. **Use corrected baselines only.** Treat ROB-129 `38.14%` WER and ROB-130 `40.86%` WER as the current frozen TEDLIUM-transfer baselines. Treat ROB-91/ROB-119 near-`100%` WER as historical evidence for the broken chunked TEDLIUM setup, not as current SSL-quality evidence.
2. **Run a full corrected-cache supervised reference probe before another SSL rerun.** ROB-128 proved one-record overfit and controls, but a full TEDLIUM corrected-cache known-good supervised encoder probe would put the `38-41%` SSL WER range on a meaningful scale.
3. **Inspect learned hidden-state weights and layer selections.** ROB-129/130 both expose six hidden states. Compare whether the useful signal comes from the same layers for paper-mask and low-mask checkpoints before deciding whether to rerun SSL.
4. **Only then choose between longer SSL and probe optimization.** If the supervised reference is strong, prioritize longer/more paper-like SSL training: more optimizer updates, potentially AdamW/Noam parity, explicit update-count checkpoint labels, and possibly more LibriSpeech-like data. If the supervised reference is also weak, prioritize probe recipe changes such as duration batching, LR/horizon sweeps, and LM/no-LM evaluation.

### Final Conclusion

ROB-98 no longer looks like a simple "BEST-RQ SSL failed" issue. The initial low-mask diagnosis was real, and the paper-style ROB-100 recipe remains the cleaner setup, but the largest observed failure came from probing on broken TEDLIUM training units. Once the TEDLIUM probe path was corrected, both the paper-mask ROB-100 checkpoint and the old low-mask ROB-70 checkpoint became CTC-usable under frozen weighted-state BiLSTM probing.

The remaining poor performance is a quality/comparability problem: corrected frozen BEST-RQ transfer is around `38-41%` TEDLIUM WER, which is useful signal but still weak. Future work should stop using the old ROB-91/119 collapse as representation-quality evidence and should compare against a full corrected supervised reference before spending another long run on SSL pretraining.
