# ROB-98 Poor SSL Performance Report

Date: 2026-05-20

Post-ROB-119 addendum: 2026-05-22

Post-ROB-125 addendum: 2026-05-22

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
- ROB-100 corrected SSL run evidence from `symphony/RESEARCH_DIARY.md`
- ROB-119 weighted BiLSTM probe summary: `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/rob119-full-weighted-bilstm-20260522T110547Z/OUTCOME.md`
- ROB-119 weighted BiLSTM probe manifest and diagnostics:
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/rob119-full-weighted-bilstm-20260522T110547Z/run_manifest.json`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-119/rob119-full-weighted-bilstm-20260522T110547Z/diagnostics/primary.jsonl`
- ROB-119 implementation branch inspected at `origin/symphony/rob-119-paper-matched-bestrq-probe`
- ROB-125 PR: https://github.com/robflynnyh/long-context-asr/pull/28
- ROB-125 known-good supervised probe summary:
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-125/rob125-full-supervised-4epoch-20260522T165541Z/OUTCOME.md`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-125/rob125-full-supervised-4epoch-20260522T165541Z/tedlium_test_results.csv`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-125/rob125-full-supervised-4epoch-20260522T165541Z/run_manifest.json`
- ROB-125 source CTC moving-window smoke:
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-125/rob125-source-ctc-moving-window-smoke-20260522T2050Z/source_ctc_moving_window_results.csv`
  - `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-125/rob125-source-ctc-moving-window-smoke-20260522T2050Z/source_ctc_moving_window_predictions.jsonl`

## Bottom Line

The strongest current explanation is not that random-codebook BEST-RQ cannot work. The open implementation shows useful downstream ASR performance with fixed random projection/codebook targets. The poor ROB-91 probes are more plausibly explained by a combination of:

1. A major SSL masking mismatch: our current config masks about 10% of stacked frames, while the paper's successful settings are much higher: Table 3's best ablation uses 12% mask starts, about 48% actual masked frames, and the main SpeechBrain/open setting uses 15% mask starts, about 60% actual masked frames.
2. A substantially non-like-for-like SSL setup: different encoder depth, subsampling, self-conditioning, optimizer, scheduler, normalization, and training horizon.
3. Undertraining or at least weak evidence of completed pretraining: the ROB-70 checkpoint labels count consumed recordings, not optimizer steps, and the run is only one epoch.
4. Probe limitations: the final ROB-91 BiLSTM probe fixed the most obvious "linear head is too weak" problem, but it still does not match the paper's hidden-state weighted-sum probing setup and it evaluates on TEDLIUM without LM support.

There is no obvious evidence that the quantizer target path is fundamentally broken after the ROB-70 fixes. The implementation uses a fixed random projection quantizer, predicts codebook classes with cross entropy on masked positions, skips empty-mask chunks, and saves the acoustic-model state for downstream loading. The main concern is that the task being learned is easier and less representation-forcing than the SpeechBrain/paper task.

Update after ROB-125: the highest-confidence new finding is that the moving-window eval and greedy CTC decoding path is not generically broken. A known-good supervised ROB-81 CTC head routed through the ROB-119-style moving-window path produced normal nonblank output on a one-record TEDLIUM smoke. However, a freshly initialized weighted-BiLSTM CTC probe trained on the frozen supervised ROB-81 encoder still collapsed badly. That means the frozen ROB-119/ROB-125 probe results are not yet a clean measure of SSL representation quality; they are confounded by a fresh random CTC-probe optimization/setup problem that must be debugged with supervised controls.

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

## Post-ROB-125 Addendum

ROB-125 ran the first recommended discriminator from the ROB-119 addendum: replace the frozen SSL encoder with a known-good supervised ASR encoder and ask whether the same general TEDLIUM CTC probe/eval route can emit words.

The known-good checkpoint was the ROB-81 supervised Floras finetune:

- `/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt`
- Prior ROB-81 TEDLIUM evidence: `8.258%` WER over `28215` test words.

ROB-125 produced two different controls with different implications:

| Control | Setup | Result | Interpretation |
| --- | --- | --- | --- |
| Fresh probe on frozen supervised encoder | Frozen ROB-81 supervised backbone, trainable weighted hidden-state sum over 3 hidden states, fresh 2-layer BiLSTM CTC probe, 4 TEDLIUM epochs, LR `1e-3` constant | `99.53%` WER, `95.79%` CER, `89.87%` deletions, `0.00%` insertions, `2859` hyp words / `28215` ref words | A fresh random weighted-BiLSTM CTC probe can still collapse even when the frozen encoder is known to contain supervised ASR information. |
| Source supervised CTC head through moving-window eval | ROB-81 checkpoint's native supervised CTC decoder wrapped into the ROB-119-style moving-window logits path plus greedy CTC scoring | One-record smoke `8.06%` WER, `4.34%` CER, `2.92%` deletions, `0.54%` insertions, `2906` hyp words / `2977` ref words | The moving-window logits path and greedy CTC decoder can emit normal words when supplied with a known-good CTC head. |

This changes the diagnosis again. ROB-125 mostly rules out a generic evaluation or greedy-decoding collapse. It does not prove that ROB-100's frozen SSL representation is useful, but it shows that the negative frozen-probe evidence is now confounded by the fresh CTC probe training path itself. The same style of random probe failed on frozen supervised features where a source-trained CTC head works.

### What ROB-125 Rules Out

1. **"The moving-window evaluator cannot produce words."** The ROB-81 source CTC head produced a normal one-record transcript through the same moving-window logits path.
2. **"Greedy CTC decoding is inherently blanking these TEDLIUM records."** The source CTC head used greedy CTC and still produced `2906` hypothesis words against `2977` reference words on the smoke record.
3. **"A known-good encoder alone guarantees the fresh weighted-BiLSTM probe will work."** It does not. Frozen supervised features plus a random weighted-BiLSTM head still yielded `99.53%` WER.

### What ROB-125 Does Not Rule Out

1. **Fresh probe optimization/setup bug or mismatch.** This is now the most direct blocker. The probe may need a different LR schedule, longer training, smaller head, source-head initialization, an overfit check, or a correction in how the weighted hidden states are exposed and normalized.
2. **Frozen SSL representation weakness.** ROB-100 may still be weak, but ROB-119 is no longer clean evidence for that by itself because the same fresh-probe family also failed on supervised features.
3. **TEDLIUM transfer and no-LM limitations.** These remain caveats, but the source CTC smoke shows they do not force near-empty output when the CTC head is already trained.
4. **ROB-126 top-layer adaptation.** ROB-126 was still running at the time of this addendum, so small supervised adaptation from ROB-100 is not yet answered here.

### Revised Current Diagnosis

The investigation now has two layers:

1. The original SSL setup did have real mismatches relative to the open BEST-RQ implementation: especially low actual mask density, self-conditioning differences, architecture/optimizer differences, and a much shorter training horizon.
2. The current downstream probe evidence is not clean enough to rank SSL representation quality ahead of probe training quality. ROB-125 narrowed a concrete problem to fresh CTC probe training on frozen features: source-trained CTC decoding works, but a newly trained weighted-BiLSTM probe does not.

The highest-value next discriminator is therefore not another full SSL rerun. It is a supervised-feature probe-debug ladder:

1. **One-record overfit check.** Use the ROB-81 supervised encoder features and train the random probe on one TEDLIUM recording until it can overfit. If it cannot overfit one recording, debug labels, CTC lengths, hidden-state selection, normalization, trainable parameter filtering, and eval checkpoint loading before interpreting any SSL probe.
2. **Head ablation on the same supervised features.** Compare the source CTC head, source CTC head initialized then trainable, random linear CTC head, and random BiLSTM head. This separates "random CTC training works" from "this large BiLSTM probe is hard to optimize."
3. **Initialization control.** Initialize the probe decoder from the ROB-81 source CTC decoder where shape-compatible, then train only the weighted sum or a small adapter. If this works while random heads fail, the issue is optimization/initialization, not moving-window decoding.
4. **Only then return to SSL probes.** Once supervised controls pass, rerun the same probe ladder on ROB-100 or wait for ROB-126. If supervised controls pass but ROB-100 still fails, the SSL representation/recipe hypothesis becomes much stronger.

This supervised-feature probe-debug ladder is tracked as child issue ROB-128: https://linear.app/robflynn/issue/ROB-128/debug-fresh-ctc-probe-training-on-known-good-supervised-features

### Current Working Answer

The best current answer is: the original BEST-RQ SSL setup was under-masked and not paper-like, ROB-100 fixed the largest masking mismatch, but the probe evidence after ROB-119 and ROB-125 is now dominated by a fresh CTC probe training failure. The eval/decoding path can produce words from a trained CTC head; the failure appears when training a new weighted-BiLSTM CTC head on frozen features. Until that supervised-control probe can overfit or otherwise produce normal output, the SSL checkpoint cannot be fairly judged from frozen-probe WER alone.

### Updated Conclusion

The initial ROB-98 conclusion should be revised, not discarded. Low actual mask density was the highest-confidence mismatch in ROB-70 and it was worth fixing. ROB-100 fixed that mismatch, but ROB-119 shows that the corrected one-epoch checkpoint still does not yield useful frozen TEDLIUM CTC representations under a substantially more paper-matched probe.

ROB-125 now says the generic moving-window eval/greedy CTC path is viable, but the fresh random weighted-BiLSTM probe can fail even on frozen supervised features. The most actionable next move is therefore a focused probe-training debug ladder on supervised features, beginning with a one-record overfit check and head/initialization ablations. ROB-126 should still be read when it completes, because top-layer unfreezing may show whether ROB-100 contains useful information after small supervised adaptation, but it should not replace the supervised probe-training control.
