# Distance Is Not Damage

## Does the KL forgetting law predict what can be recovered after fine-tuning?

**Research proposal — revised August 2026**

> **Contribution in one sentence.** Prior work shows that new-task KL predicts the *immediate magnitude* of forgetting; this project tests whether it also predicts *recovery*, and whether recovery requires a second axis determined by externally supplied information and the geometry of the update, beginning with a preregistered full-fine-tuning versus rank-8 LoRA contrast.

## Abstract

Fine-tuning can reduce a model's performance on capabilities learned earlier. Shenfeld, Pari, and Agrawal (2025) show that the forward KL divergence from the base policy, evaluated on the new task, predicts the magnitude of this drop in a controlled setting ($R^2=0.96$) and remains predictive in LLM experiments ($R^2=0.71$). That result says how much performance is lost at the end of fine-tuning. It does not say whether the loss reflects inaccessible knowledge that can be cheaply restored or overwritten knowledge that must be learned again.

This proposal asks whether the KL forgetting law extends from *damage now* to *recovery later*. We first reproduce the KL–forgetting relationship on ParityMNIST/FashionMNIST while crossing two independent factors: training objective (off-policy SFT or on-policy learning) and update parameterization (full fine-tuning or rank-8 LoRA). We then run a controlled sequential-learning experiment in which the new task's prompts, output lengths, token marginals, and training objective are held fixed while the prompt–target association is varied from base-derived to randomly permuted. The latter mapping is information the base model cannot know and therefore must be acquired from the fine-tuning data. We independently vary capacity pressure and update geometry, sweep learning rates and steps to create overlapping KL and immediate-forgetting ranges, and attach the same preregistered recovery protocol to every checkpoint.

The primary outcome is recovery advantage over a matched model that never learned the old task. Candidate predictors are new-task KL, immediate forgetting, new-task compression advantage, old-task code-length loss, fixed and freshly fitted probes, representational drift, update-subspace geometry, and local sharpness. The primary analysis tests out-of-sample prediction; the randomized information-load manipulation and full-versus-low-rank contrast test different parts of the mechanism. The result will establish either that KL is a sufficient practical signal for both forgetting and recovery, or that checkpoints at the same behavioral distance can differ because they wrote different information or took different geometric paths.

## 1. The problem

Suppose a model's score on an earlier task falls from 80% to 30% after fine-tuning. There are at least two operationally different explanations:

1. **Access loss:** useful internal structure remains, but the model no longer maps the task to the right behavior. A small alignment intervention may restore performance.
2. **Content loss:** task-specific information no longer provides a measurable learning advantage. Restoring performance requires reacquiring it from data.

The observed 50-point drop does not distinguish these cases. Neither does the fact that the final checkpoint moved far from the base model. The distinction matters in opposite directions for continual learning and unlearning: a practitioner wants ordinary forgetting to be cheap to reverse, while an unlearning claim is weak if a small intervention restores the target behavior.

The proposed project is deliberately narrower than a general theory of forgetting. It asks one predictive question and two mechanistic questions:

- **Predictive:** Given a fine-tuned checkpoint, does new-task KL forecast its subsequent recovery curve?
- **Mechanistic:** At matched KL and immediate forgetting, does externally supplied information load change recovery?
- **Geometric:** At matched new-task performance, KL, and immediate forgetting, does a rank-8 update constraint change what remains decodable and how cheaply the old task recovers?

## 2. What is known, and what is missing

| Question | Current evidence | What remains open here |
|---|---|---|
| Does new-task KL predict immediate forgetting? | Yes in the settings studied by [Shenfeld et al. (2025)](https://arxiv.org/abs/2509.04259): $R^2=0.96$ on ParityMNIST and $R^2=0.71$ in their LLM experiments. | Replication is required before extending the result. |
| Can equal behavioral loss differ in recoverability? | Yes. [Zheng et al. (2025)](https://arxiv.org/abs/2501.13453) demonstrate rapid recovery after apparent forgetting; [Xu et al. (2025)](https://arxiv.org/abs/2505.16831) map reversible and budget-resistant regimes in unlearning. | The ordinary-fine-tuning setting has not been connected to the KL law. |
| Are there pre-recovery diagnostic candidates? | Yes. Xu et al. find representation drift predictive under fixed unlearning protocols; [Fan et al. (2025)](https://arxiv.org/abs/2502.05374) motivate smoothness as a defense against relearning attacks. | No head-to-head, cross-task comparison with KL and information measures has established a general predictor. |
| Are standard output metrics sufficient? | No. [OpenUnlearning](https://arxiv.org/abs/2506.12618) shows that commonly used metrics vary in faithfulness and robustness; [REBEL](https://arxiv.org/abs/2602.06248) recovers behavior missed by static prompts. | A recovery outcome should not be defined by a single prompt or one post-fine-tuning score. |
| Does information acquired during the new task predict recovery of the old task? | Compression-based work can measure memorization and behavior change in bits ([Morris et al., 2025](https://arxiv.org/abs/2505.24832); [Tan et al., 2026](https://arxiv.org/abs/2607.21351)). | This has not been tested as a predictor of recovery after ordinary fine-tuning. |
| Does LoRA rank determine forgetting or recovery? | [Biderman et al. (2024)](https://arxiv.org/abs/2405.09673) find that standard low-rank LoRA often learns less and forgets less than full fine-tuning; a recent preprint links LoRA forgetting to task-gradient subspace angles ([Steele, 2026](https://arxiv.org/abs/2603.02224)). | Existing evidence does not establish that low rank is intrinsically “access loss,” that full fine-tuning is “overwrite,” or that rank predicts recovery after KL and immediate damage are matched. |

The gap is therefore not “nobody has studied reversibility.” That would be false. The gap is the missing crossing: **no cited work tests whether the new-task KL law predicts recovery after ordinary fine-tuning, or whether controlled information load and update geometry add predictive signal at matched KL and immediate damage.**

## 3. How the cited literatures fit together

### 3.1 KL supplies the baseline predictor

Shenfeld et al. provide the empirical law this project tries to extend. Their result also supplies the first testbed, metric definition, hyperparameter-sweep strategy, oracle policy construction, and competing-predictor format. Their on-policy result is relevant because it explains how different training procedures populate different parts of the KL range; it is not evidence by itself about recovery.

Their distillation result should be interpreted narrowly. An SFT student trained on outputs from an RL teacher matched the teacher's measured learning–forgetting trade-off. This supports a final-policy account of *immediate forgetting on the measured distributions*. It does not establish that the two models are identical distributions everywhere, nor that their recovery dynamics must match.

### 3.2 Continual learning and unlearning supply the outcome

Zheng et al. show that severe behavioral forgetting can coexist with strong recovery after limited retraining, motivating access loss as a real phenomenon. Xu et al. formalize reversibility under a bounded relearning protocol and show that representation diagnostics can separate regimes within their unlearning setup. These papers justify measuring an entire recovery curve rather than treating the terminal accuracy drop as erasure.

[Davari et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.html) further show why behavioral and representational forgetting should be separated with linear probes. This proposal makes that distinction explicit: a probe fitted once at $\theta_0$ detects whether the old coordinate/readout remains stable, while a fresh probe fitted at each checkpoint detects whether task-A labels remain linearly decodable after the representation moves. A fresh probe is allowed to learn from A labels, so it measures usable information for that probe class—not zero-shot access by the deployed model and not literal bits stored in the weights.

Unlearning evidence is transferred cautiously. Unlearning objectives intentionally suppress a target behavior; ordinary fine-tuning optimizes a different task. A predictor that works in unlearning may fail in sequential fine-tuning. Testing that transfer is part of the contribution, not an assumption.

### 3.3 Compression supplies a measurement and a task-design principle

For a target dataset $D=\{(x_i,y_i)\}$, define the behavior-write quantity

$$
W(D;\theta_0,\theta)=\sum_i \left[-\log_2 p_{\theta_0}(y_i\mid x_i)+\log_2 p_{\theta}(y_i\mid x_i)\right].
$$

This is a change in conditional code length on specified targets. On uniformly random targets, where generalization is ruled out by construction, compression advantage is evidence of memorization. On natural text, $W$ mixes memorization, generalization, and changes in output convention. It is therefore a **behavioral code-length measure**, not a literal census of bits physically located in particular weights. Tan et al.'s excess statistic, which subtracts a size-matched held-out change, helps isolate training-specific information but does not remove every semantic or surface-form confound.

The distinction is load-bearing. The experiment separately measures:

- $B_{\text{new}}$: excess compression advantage on the new task, calibrated with random associations;
- $L_{\text{old}}$: increase in code length on old-task targets after fine-tuning;
- $C_{\text{recovery}}$: recovery behavior under a protocol that does not use either code-length quantity as its label.

### 3.4 What task design can and cannot guarantee

The input data are the cleanest available lever because the internal mechanism cannot be assigned directly.

| Fine-tuning content | External target information introduced? | Expected training-specific write | Role in the experiment |
|---|---:|---:|---|
| Random prompt–value associations or random strings | **Yes, by construction** | High if successfully learned | Positive control for deposition |
| Targets sampled or selected from the base model | No information outside the base/data-generation system | Often low, but not guaranteed to be zero | Sharpening/reweighting control |
| Deterministic format or style transformation | Little semantic information; the rule itself still has a description length | Near zero after held-out subtraction if the rule generalizes | Access/format control |

[Huang et al. (2025)](https://arxiv.org/abs/2412.01951) show why self-improvement cannot create information absent from the model and formalize self-improvement as sharpening. This constrains *epistemic novelty*. It does **not** imply that self-training makes zero parameter changes or that $W(D)$ must be exactly zero: a model can sharpen, re-encode, or memorize a sampled transcript without acquiring an externally supplied fact. Likewise, on-policy RL can only learn an exact random secret if it samples informative reward-bearing outputs; failure to do so is a coverage limitation, not a clean recovery comparison.

The primary causal contrast therefore uses SFT in every cell and changes only the prompt–target association. SFT versus RL is a secondary method comparison on tasks both methods can solve.

### 3.5 Low rank constrains the update; it does not name the mechanism

LoRA and SFT are not competing categories. SFT specifies a loss and source of targets; LoRA specifies which parameter updates are allowed. The experiment must therefore cross **objective/data** with **update parameterization**, rather than call one cell “SFT” and another “LoRA.”

For a pretrained matrix $W_0$, [Hu et al. (2021)](https://arxiv.org/abs/2106.09685) define a frozen-base update

$$
W = W_0 + \Delta W,
\qquad
\Delta W = \frac{\alpha}{r}BA,
\qquad
\operatorname{rank}(\Delta W)\le r.
$$

This is an additive low-rank constraint. It is not generally an orthogonal rotation, and it does not imply that the adapter can only memorize. Conversely, full fine-tuning permits high-rank updates but does not imply that useful content was overwritten. “Rotation” will be used here only as shorthand for an **access/reorientation hypothesis**—old information remains decodable but the deployed readout or task alignment changes—not as a claim that the learned weight matrices are literal rotation matrices. [Zheng et al. (2025)](https://arxiv.org/abs/2501.13453) provide evidence that early apparent forgetting can reflect task-alignment disruption; [Jin et al. (2026)](https://arxiv.org/abs/2605.10973) show that penalizing changes in projected pretrained singular subspaces can improve the in-domain/OOD trade-off. Neither result establishes the proposed LoRA/full dichotomy.

The empirical literature gives competing predictions. [Biderman et al. (2024)](https://arxiv.org/abs/2405.09673) find that standard LoRA settings often learn less of the target task and forget less of the source domain than full fine-tuning, while many learning–forgetting points lie on similar Pareto frontiers. Thus lower forgetting can be under-adaptation rather than a distinct preservation mechanism. [Tan et al. (2026)](https://arxiv.org/abs/2607.21351) find that writable bits depend strongly on placement and pretrained structure and need not grow smoothly with rank. A recent single-author preprint reports that rank effects on forgetting interact with the principal angles between task-gradient subspaces ([Steele, 2026](https://arxiv.org/abs/2603.02224)); this is treated as a candidate covariate to test, not an established law.

The recoverability question is therefore sharper than “does LoRA forget less?” It is:

> Among checkpoints with the same task-B performance, new-task KL, and immediate task-A damage, does rank-8 LoRA leave a different recovery curve from full fine-tuning, and is that difference explained by written bits, probe behavior, or update-subspace geometry?

Disabling a LoRA adapter is a useful implementation sanity check because it should exactly restore the frozen base. It is **not** a valid recovery result: it also deletes the learned task-B behavior. Every reported recovery curve keeps the task-B update active and measures task-B retention alongside task-A restoration.

## 4. Research questions and preregistered hypotheses

### RQ1 — Replication

Does forward KL on the new task reproduce the published relationship with immediate forgetting?

**H1.** Across methods, schedules, learning rates, and checkpoints in ParityMNIST, new-task KL explains most variation in FashionMNIST forgetting and outperforms weight-distance baselines.

### RQ2 — Prediction

Does new-task KL predict recovery after ordinary fine-tuning?

**H2.** KL alone predicts some recovery variation because it predicts immediate damage, but a model that also observes immediate forgetting and information/geometry measures will improve held-out recovery prediction.

The strict null is useful: if grouped cross-validation shows no material gain beyond KL and immediate forgetting, the KL law extends operationally to recovery in the tested domain.

### RQ3 — Mechanism

Does externally supplied information load alter recovery at matched KL and immediate forgetting?

**H3.** The effect of information load is conditional on capacity pressure. Novel associations should make recovery harder primarily as the new dataset approaches the measured writable capacity of the trainable parameterization. A main effect of random targets without an information-load × capacity-pressure interaction is insufficient evidence for capacity displacement.

### RQ4 — Rank and update geometry

Does rank-8 LoRA change recovery relative to full fine-tuning after matching task-B performance, new-task KL, and immediate task-A forgetting?

**H4a.** At matched task-B performance, rank-8 LoRA will usually show less immediate task-A forgetting than full fine-tuning. This tests whether a published learning–forgetting pattern transfers to the present setting and is not yet the recovery contribution.

**H4b.** After additionally matching KL and immediate forgetting, update parameterization will retain incremental predictive value for the recovery curve if endpoint behavioral distance is not sufficient. The directional mechanism is deliberately tested rather than assumed: an access/reorientation result would combine a larger fixed-versus-fresh probe gap with preserved old-task code length and positive recovery advantage; a rewrite result would combine fresh-probe loss, old-task code-length loss, and recovery approaching the never-knew baseline.

### RQ5 — Competing signatures

Which pre-recovery signal generalizes across task templates and model scales?

- **Access/interference signature:** large immediate behavioral loss, little old-task code-length loss, representational reorientation, and recovery from task-format cues that contain no old facts.
- **Rewrite/capacity signature:** high new-task deposition under high capacity pressure, old-task code-length loss, and little recovery advantage over a matched model that never learned the old task.
- **Geometry signature:** recovery tracks update rank/spectrum, task-gradient subspace angle, or preregistered sharpness after controlling for KL, immediate forgetting, information load, and task performance.

These are signatures, not mutually exclusive ontologies. A model can experience access loss and content loss in the same run.

## 5. Experimental design

### 5.1 Phase 0: reproduce the KL law

Reproduce the ParityMNIST/FashionMNIST setup from Shenfeld et al.:

- 3-layer MLP with dimensions $785\rightarrow512\rightarrow256\rightarrow10$;
- flattened image plus task indicator ($+1$ for ParityMNIST, $-1$ for FashionMNIST);
- joint pretraining on 500 examples per task;
- ParityMNIST SFT labelings, on-policy objectives, and the base-conditioned oracle target distribution;
- 15 log-spaced learning rates from $3\times10^{-6}$ to $10^{-3}$, two schedules, and one/two epochs;
- float32 training, checkpoint-level metrics, and at least three seeds for the confirmatory run.

Beginning in Week 1, cross the two primary objectives with two update parameterizations:

| Factor | Primary levels | Purpose |
|---|---|---|
| Objective/data path | SFT-1; 1–0 REINFORCE | Reproduce the off-policy/on-policy contrast. |
| Update parameterization | Full fine-tuning; LoRA rank 8 | Test whether constrained update geometry changes forgetting at comparable learning. |

For the MLP pilot, apply LoRA to all three linear layers, freeze their pretrained weights and biases, set $r=8$, $\alpha=8$, and dropout to zero. The $\alpha/r=1$ scaling and zero-initialized LoRA output make the initial deployed function exactly equal to $\theta_0$. Sweep learning rates separately for full fine-tuning and LoRA; using one shared “best” learning rate would confound update geometry with avoidable under-training. SFT-2, oracle SFT, GRPO, and GRPO-KL are robustness cells after the four-cell core is verified.

Week 1 records task-B accuracy and KL, task-A forgetting and label code length, effective-weight distance, CKA, and both fixed and fresh task-A linear probes. Because the label set is only ten familiar classes, its code length is a behavioral confidence diagnostic, not a writable-capacity estimate. Phase 0 does not support a bits-per-parameter claim.

**Gate P0.** Continue only if (i) the SFT/on-policy learning–forgetting separation is qualitatively reproduced, (ii) the fitted KL–forgetting relationship is stable across seeds, and (iii) full and rank-8 runs have common support in task-B performance and KL. The target is not mechanically $R^2=0.96$; confidence intervals and failure analysis are reported. If condition (iii) fails, the LoRA learning-rate/step grid is expanded before any rank comparison is interpreted.

ParityMNIST is a replication harness, not the main bits experiment. Its ten-label output space and familiar labels do not create enough controlled information load to adjudicate capacity displacement.

### 5.2 Phase 1: paired information-load experiment

Use a small decoder model whose task-B update is either full fine-tuning or all-linear-layer LoRA. Rank 8 versus full is the primary parameterization contrast inherited from Phase 0; additional ranks and placements are ablations, not separate headline experiments. This makes writable capacity calibratable while retaining a direct high-rank comparison.

#### Checkpoints and tasks

1. Start with $\theta_{\text{pre-A}}$, which has never seen synthetic task A.
2. Teach task A, a set of synthetic key–value facts, producing base checkpoint $\theta_0$.
3. Fine-tune $\theta_0$ on task B to obtain $\theta_B$.
4. In parallel, apply the same task-B procedure to $\theta_{\text{pre-A}}$ to obtain a matched **never-knew-A control**.

The never-knew control is essential. Fast relearning from $\theta_B$ is evidence of retained information only if it is faster than learning A from a model with the same B-training history that never knew A.

#### Randomized information-load manipulation

Build a base-derived set of prompts and target strings. Randomly permute a preregistered fraction $\lambda\in\{0,.25,.5,1\}$ of target assignments across prompts. This preserves:

- the prompt set;
- the exact target strings;
- target length and token-frequency marginals;
- model architecture, optimizer, and loss;
- dataset size and number of exposures.

It changes only how much prompt–target association the base cannot know. Fully random token strings remain a calibration condition, not the sole treatment, because natural versus random strings would otherwise confound information load with surface form and initial likelihood.

Add a deterministic formatting cell as a low-information positive-learning control. Measure initial loss and gradient statistics in every cell rather than assuming the construction matches difficulty perfectly.

#### Capacity pressure

For each update parameterization, and for every LoRA placement/rank ablation, estimate writable capacity using the random-string protocol of Morris et al. and Tan et al. Define

$$
\rho = \frac{\text{entropy of novel task-B associations}}{\text{measured writable capacity at the declared training budget}}.
$$

Run at several $\rho$ bands below and around one. Capacity claims require this axis: writing novel facts into a largely empty parameterization does not imply that old facts must be displaced.

#### Creating overlap

Sweep learning rate and training steps independently within every $(\lambda,\rho,\text{parameterization})$ cell. Retain a common-support region in which cells overlap on:

- new-task performance;
- new-task forward KL;
- immediate old-task forgetting.

Matched comparisons are made only inside this overlap. Extreme runs remain useful for curve fitting but do not support the matched causal claim.

### 5.3 Secondary algorithm comparison

Compare SFT, 1–0 REINFORCE, and GRPO only on task-B variants whose correct outputs have adequate support under $\theta_0$. This tests whether algorithm adds recovery signal after conditioning on final-policy measurements.

Do not use an RL agent's failure to learn an unguessable random code as evidence that RL “writes no bits.” In that condition the reward supplies no useful gradient until the code is sampled, so task performance is not matched.

### 5.4 Measurements at every checkpoint

Measure before any recovery intervention:

| Construct | Primary measure | Notes |
|---|---|---|
| New-task distribution shift | $D_{\mathrm{KL}}(\pi_0\|\pi_B)$ on fixed task-B prompts | Match Shenfeld et al.; also report reverse KL. |
| Immediate forgetting | Task-A performance drop from $\theta_0$ to $\theta_B$ | Condition on this when asking about recovery. |
| New information acquired | $B_{\text{new}}$: training-specific compression advantage on B | Ground-truth interpretation is strongest for permuted/random associations. |
| Old behavioral information lost | $L_{\text{old}}$: code-length increase on canonical and paraphrased A targets | Report canonical, max-over-paraphrases, and content-token-only variants. |
| Old information decodability | Fixed and fresh linear probes on frozen A representations | The fixed $\theta_0$ probe tests coordinate/readout stability; a fresh checkpoint-specific probe tests linear decodability. Both require A labels and are scientific diagnostics, not deployable no-A signals. |
| Representation change | Layerwise CKA/CKNNA on A, B, and neutral prompts | Geometry is a candidate predictor, not proof of content. |
| Update geometry | Effective-update rank/singular spectrum and principal angles between A/B gradient subspaces | LoRA's declared rank is a constraint; the realized spectrum and task overlap are measured rather than inferred. |
| Local geometry | Normalized perturbed-loss gap plus a Hessian trace/eigenvalue estimate | Predeclare normalization because raw sharpness is parameterization-sensitive. |
| Update baselines | L2, Fisher-weighted L2, cosine update alignment | Included to connect to the published predictor ablation. |

### 5.5 Recovery protocol and outcome

“Irreversible” is not an observable finite-budget property. The proposal therefore uses **budget-resistant** and defines recovery relative to explicit compute and data.

Each terminal checkpoint receives three recovery arms:

1. **Cue-only recovery:** task-format/alignment examples containing no task-A facts. Success is strong evidence of access loss.
2. **Direct relearning:** a stratified subset of task-A examples. This measures sample efficiency but is interpreted only relative to the never-knew-A control.
3. **Ceiling recovery:** the largest preregistered A-data and compute budget. This distinguishes slow recovery from an early plateau.

All recovery arms preserve the task-B update and report both A recovery and B retention. For LoRA checkpoints, “disable the B adapter” is logged only as a sanity bound and is excluded from recovery outcomes because it restores A by discarding B.

For each arm, evaluate a budget ladder in examples, tokens, and optimizer steps. Use the best result from a small preregistered recovery-learning-rate grid so that a checkpoint is not labeled resistant merely because one learning rate was unsuitable.

Primary recovery outcomes are:

- **Recovery AULC:** area under the task-A recovery curve on a log-budget axis;
- **Recovery advantage:** AULC from $\theta_B$ minus AULC from the matched never-knew-A control;
- **Recovery ceiling:** best task-A score within the maximum budget;
- **$C_{90}$:** minimum budget restoring 90% of the lost score, reported as a secondary, censored outcome with 80% and 95% sensitivity analyses.

Recovery advantage, not raw recovery speed, is the closest operational measure of retained task-A information.

### 5.6 Rank and geometry ablations

The main full-versus-rank-8 comparison changes both the dimension and the factorized shape of the trainable update. It is the practical estimator comparison, not a pure causal estimate of rank alone. Use the following staged ablations to identify what drives any difference:

| Ablation | Comparison | Question isolated |
|---|---|---|
| Primary | Full fine-tuning vs all-linear LoRA $r=8$ | Does the deployed update parameterization change recovery at matched endpoints? |
| Rank | LoRA $r\in\{2,8,32\}$ | Is any effect monotonic or thresholded in rank? Run only after the primary contrast. |
| Placement | All linear layers vs MLP-only vs attention-only | Is writable capacity/forgetting driven by where updates are allowed? Transformer phase only. |
| Parameter budget | LoRA vs a matched-dimensional random update subspace | Is the effect specific to factorized low rank or mostly trainable dimension? |
| Learning control | Independently tuned learning-rate/step grids; matched task-B performance | Does “LoRA forgets less” survive equal learning rather than reflect underfitting? |
| Geometry | High- vs low-A/B gradient-subspace-angle task pairs | Does rank matter only when task update subspaces overlap? |
| Bias/scaling | Frozen vs trainable bias; fixed $\alpha/r$ | Are implementation details masquerading as rank effects? |

These ablations are ordered to control scope. Week 1 runs only the primary full-versus-$r=8$ comparison, with all linear layers targeted and separate learning-rate sweeps. Phase 1 adds $r\in\{2,8,32\}$ if the primary contrast has common support and a detectable effect. Placement and matched-dimensional random-subspace controls are LLM-stage mechanism checks.

### 5.7 Analysis plan

Separate prediction from causal inference.

#### Predictive analysis

Fit the following nested predictors of recovery AULC:

1. immediate forgetting only;
2. immediate forgetting + new-task KL;
3. baseline + $B_{\text{new}}$ and $L_{\text{old}}$;
4. baseline + fixed/fresh probe gap and representation measures;
5. baseline + parameterization, realized update spectrum, task-subspace angle, and sharpness;
6. all preregistered predictors.

Compare grouped out-of-sample $R^2$, rank correlation, and calibration. Hold out entire task templates and random-mapping seeds; later hold out a model scale. Checkpoints from the same training run are never split across train and test folds. Report bootstrap confidence intervals and the incremental performance of each predictor family, not only in-sample fits.

#### Causal analysis

Estimate the randomized effects of $\lambda$, $\rho$, parameterization, and their preregistered interactions within the common-support region, controlling for task performance, KL, and immediate forgetting. The information-load × capacity-pressure interaction is the confirmatory capacity test. The rank-8/full coefficient and parameterization × task-subspace-angle interaction test whether update geometry adds information beyond the behavioral endpoint. Mediation through measured $B_{\text{new}}$, $L_{\text{old}}$, or probes is exploratory because those quantities are post-treatment variables.

#### Falsification checks

- Held-out random associations must have near-zero training-specific compression advantage.
- Surface-form variants must agree in sign; otherwise the code-length instrument is format-dominated.
- Predictors must transfer to held-out mappings/tasks; checkpoint interpolation alone is insufficient.
- Results must survive per-run aggregation so dense checkpoint logging does not inflate the effective sample size.
- Full and LoRA cells must overlap in task-B performance, KL, and immediate forgetting; otherwise the result is a Pareto comparison, not a matched-recovery comparison.

### 5.8 Phase 2 / future work: no-old-data recovery triage

The primary project asks whether checkpoint signals predict recovery under a standardized protocol. A natural second-stage question is more decision-oriented:

> **Can signals measured without old-task examples or additional model training predict whether a checkpoint is recoverable, which recovery treatment to use, and the approximate data/compute budget required?**

This is possible as a learned meta-prediction problem, with an important information boundary. At deployment, the system must know which capability is being discussed through a task identifier or coarse metadata, and it may use the base checkpoint, fine-tuned checkpoint, task-B data/statistics, and training logs. It does not receive task-A examples or labels. Without even a task identity or target definition, “recover this capability” is ill-posed. Without task-A evaluation data, the system can recommend an intervention but cannot certify afterward that 90% recovery was actually achieved.

The training of the triage predictor is also separate from its deployment. During research, we create supervision by running recovery experiments on many checkpoints. At deployment, applying the fitted predictor requires no model fine-tuning and no task-A data.

#### Recovery treatments

For every research checkpoint, estimate a recovery curve for each treatment $t$ over data/compute budget $b$:

$$
R_t(b\mid\theta).
$$

| Treatment | Contains task-A facts? | What it tests |
|---|---:|---|
| Cue/format examples | No | Can task alignment be restored without reintroducing content? |
| Domain-related examples | No | Can related behavior reactivate the capability? |
| Direct task-A examples | Yes | How efficiently can task-specific content be recovered? |
| Full replay | Yes | What recovery ceiling is reachable within the declared compute budget? |

The treatment menu must be defined in advance. The predictor can choose among known intervention classes; it cannot infer or invent unavailable task-A facts from checkpoint statistics alone.

#### Two predictor versions

The **deployable predictor** uses only signals available without task-A examples or labels:

- forward/reverse KL on task B;
- task-B performance, training dynamics, and task metadata;
- weight-update and optimizer statistics;
- update parameterization, declared rank, effective update spectrum, and task-B/neutral-data subspace statistics;
- sharpness measured on B;
- representation drift on neutral data and model-architecture metadata.

The **scientific upper-bound predictor** additionally receives immediate A forgetting, old-task code-length loss, and A-label probes. It is not a deployable product. Its role is to quantify how much predictive power is lost when old-task access is forbidden and to test whether bits/probes explain failures of the deployable signals.

#### Decision rule

The fitted model predicts the recovery response surfaces $\widehat{R}_t(b\mid\theta)$ and their uncertainty, then recommends

$$
(t^*,b^*)
=
\arg\min_{t,b}\operatorname{Cost}(t,b)
$$

subject to

$$
P\!\left(R_t(b)\geq 90\%\text{ recovery}\right)\geq0.9.
$$

The output should be a calibrated range and fallback policy, not false precision. For example:

> “Estimated access loss probability: 0.87. Try 32–64 cue-only examples. If that fails, use 200–300 direct task examples. Abstain and run an explicit recovery audit if neither budget is available.”

Evaluate this policy by holding out entire task families and model scales, measuring coverage of its budget intervals, treatment-selection regret relative to the empirically cheapest successful treatment, and abstention quality under distribution shift.

#### Scope decision

Predicting $C_{90}$ for one fixed direct-relearning protocol is already part of the primary contribution. Predicting the cheapest treatment among cue-only, domain-related, direct replay, and full replay multiplies the recovery matrix and requires substantially more task diversity. It is therefore Phase 2: begin only if Phase 1 predicts recovery on held-out task templates rather than merely interpolating checkpoints from familiar runs.

## 6. What each possible result would mean

| Result | Supported conclusion | Conclusion not licensed |
|---|---|---|
| KL predicts held-out recovery and other measures add no stable signal | KL is an adequate operational predictor of recovery in the tested regimes. | KL is the universal mechanism of forgetting. |
| Information load × capacity pressure changes recovery at matched KL/damage | Capacity-constrained deposition contributes to budget-resistant forgetting. | Every lost capability was physically overwritten. |
| Rank-8 LoRA and full fine-tuning have different recovery at matched learning/KL/damage | Update parameterization contains recovery-relevant information beyond the measured behavioral endpoint. | Low rank is inherently “rotation,” full fine-tuning is inherently “overwrite,” or rank caused the effect without the staged ablations. |
| Fixed probe fails while a fresh probe and old-task code length remain strong | The old readout/coordinate system is disrupted while linearly usable A information remains. | All of task A is intact, or a probe-free deployed model can access it. |
| Fresh probe, old-task code length, and recovery advantage all collapse | The checkpoint shows convergent evidence of content-level loss under the declared assays. | The weights contain literally zero information about A. |
| Cue-only recovery succeeds with little old-task code-length loss | Access loss explains a meaningful portion of the behavioral drop. | Internal representations are unchanged. |
| Sharpness generalizes across tasks after all controls | Local geometry is a useful recovery predictor beyond KL. | The unlearning mechanism transfers unchanged to all fine-tuning. |
| No recovery variation remains after matching immediate damage | This testbed cannot distinguish recovery mechanisms. | Recovery is always determined by KL. |

## 7. Kill criteria and scope boundaries

- **K1 — Replication failure:** no stable KL–forgetting relationship in Phase 0. Stop extension claims and diagnose the reproduction.
- **K2 — No common support:** information-load cells do not overlap in KL, new-task performance, and immediate forgetting. Do not report matched causal effects; redesign the task or sweep.
- **K3 — No capacity pressure:** all runs have $\rho\ll1$. No capacity-displacement conclusion is possible.
- **K4 — Instrument failure:** code-length results reverse across canonical/paraphrased/content-token scoring. Drop the bits predictor as a content measure; retain the randomized treatment and other predictors.
- **K5 — Recovery-protocol failure:** known-A and never-knew-A controls have indistinguishable recovery even before B. The recovery assay lacks power and must be repaired.
- **K6 — Parameterization support failure:** full and rank-8 cells cannot be matched on task-B learning, KL, and immediate forgetting after expanding their learning-rate/step grids. Report separate Pareto frontiers; do not claim that rank changes recoverability at an equal endpoint.

The toy experiments can establish identifiability and measurement behavior. Any claim about foundation-model post-training requires a reduced-density LLM replication. Any claim about unlearning remains a transfer hypothesis until tested on unlearning objectives.

## 8. Contributions

If completed, the project contributes:

1. **A recovery extension of the KL forgetting law:** the first direct test of whether new-task KL predicts recovery after ordinary fine-tuning.
2. **A controlled information-load intervention:** output marginals are preserved while prompt–target novelty is randomized, separating information source from obvious surface-form differences.
3. **A recovery assay with a never-knew baseline:** rapid relearning is credited to retained information only when it beats an appropriately matched control.
4. **A head-to-head predictor benchmark:** KL, behavioral code length, representation drift, sharpness, and weight metrics compete under grouped held-out evaluation.
5. **A parameterization-aware recovery test:** full fine-tuning and rank-8 LoRA are compared from Week 1, then separated from learning level, rank, placement, and task-subspace overlap through staged ablations.
6. **A bridge with explicit limits:** unlearning contributes recovery protocols and candidate diagnostics; ordinary fine-tuning supplies the target domain; neither literature is treated as proof for the other.

## 9. Execution plan

| Time | Work | Decision gate |
|---|---|---|
| Week 1 | Run the four-cell objective × parameterization pilot: SFT-1 and 1–0 REINFORCE, each with full fine-tuning and all-layer LoRA $r=8$; verify exact zero-step equivalence, trainable-parameter accounting, effective-weight metrics, bits, CKA, and fixed/fresh probes. | P0 harness and LoRA correctness |
| Weeks 2–3 | Full Phase-0 sweep with independently tuned full/LoRA grids and at least three seeds; estimate common support and preregister the recovery protocol. | K1, K6 |
| Weeks 4–5 | Build paired A/B synthetic task, never-knew control, and random-association calibration. | K5 |
| Weeks 6–7 | Information-load × capacity-pressure sweep and recovery arms. | K2–K4 |
| Weeks 8–9 | Grouped predictive analysis, ablations, and held-out-task validation. | Toy-stage claim |
| Thereafter | Reduced-density LLM validation and, separately, unlearning transfer. | External validity |
| Phase 2 / future | Fit and validate the no-old-data treatment-and-budget triage policy across held-out task families. | Deployable decision support |

## References

- Aghajanyan, A., Gupta, S., & Zettlemoyer, L. (2021). [*Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*](https://arxiv.org/abs/2012.13255).
- Biderman, D., et al. (2024). [*LoRA Learns Less and Forgets Less*](https://arxiv.org/abs/2405.09673).
- Davari, M., et al. (2022). [*Probing Representation Forgetting in Supervised and Unsupervised Continual Learning*](https://openaccess.thecvf.com/content/CVPR2022/html/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.html). CVPR 2022.
- Fan, C., et al. (2025). [*Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond*](https://arxiv.org/abs/2502.05374).
- Hu, E. J., et al. (2021). [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
- Huang, A., et al. (2025). [*Self-Improvement in Language Models: The Sharpening Mechanism*](https://arxiv.org/abs/2412.01951). ICLR 2025.
- Jin, H., et al. (2026). [*Rotation-Preserving Supervised Fine-Tuning*](https://arxiv.org/abs/2605.10973). Preprint.
- Morris, J. X., et al. (2025). [*How Much Do Language Models Memorize?*](https://arxiv.org/abs/2505.24832).
- OpenUnlearning / Dorna, V., et al. (2025). [*Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics*](https://arxiv.org/abs/2506.12618).
- Rybak, P., et al. (2026). [*REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop*](https://arxiv.org/abs/2602.06248).
- Shenfeld, I., Pari, J., & Agrawal, P. (2025). [*RL's Razor: Why Online Reinforcement Learning Forgets Less*](https://arxiv.org/abs/2509.04259).
- Steele, B. (2026). [*Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation*](https://arxiv.org/abs/2603.02224). Preprint.
- Tan, K., Du, H., & Feng, Y. (2026). [*How Many Bits Can an Adapter Write? Measuring the Capacity and Memorization of Parameter-Efficient Fine-Tuning*](https://arxiv.org/abs/2607.21351).
- Xu, X., et al. (2025). [*Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs*](https://arxiv.org/abs/2505.16831).
- Zheng, J., Cai, X., Qiu, S., & Ma, Q. (2025). [*Spurious Forgetting in Continual Learning of Language Models*](https://arxiv.org/abs/2501.13453). ICLR 2025.
