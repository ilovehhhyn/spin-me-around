# Distance Is Not Damage

## Does the KL forgetting law predict what can be recovered after fine-tuning?

**Research proposal — revised August 2026**

> Prior work shows that new-task KL predicts the *immediate magnitude* of forgetting; this project tests whether it also predicts *recovery*, and whether recovery requires a second axis determined by externally supplied information and the geometry of the update, beginning with a preregistered full-fine-tuning versus rank-8 LoRA contrast.

## Abstract

Fine-tuning can reduce a model's performance on capabilities learned earlier. Shenfeld, Pari, and Agrawal (2025) show that the forward KL divergence from the base policy, evaluated on the new task, predicts the magnitude of this drop in a controlled setting ($R^2=0.96$) and remains predictive in LLM experiments ($R^2=0.71$). That result says how much performance is lost at the end of fine-tuning. It does not say whether the loss reflects inaccessible knowledge that can be cheaply restored or overwritten knowledge that must be learned again.

This proposal asks whether the KL forgetting law extends from *damage now* to *recovery later*. We first reproduce the KL–forgetting relationship on ParityMNIST/FashionMNIST while crossing two independent factors: training objective (off-policy SFT or on-policy learning) and update parameterization (full fine-tuning or rank-8 LoRA). We then run a controlled sequential-learning experiment in which the new task's prompts, output lengths, token marginals, and training objective are held fixed while the prompt–target association is varied from base-derived to randomly permuted. The latter mapping is information the base model cannot know and therefore must be acquired from the fine-tuning data. We independently vary capacity pressure and update geometry, sweep learning rates and steps to create overlapping KL and immediate-forgetting ranges, and attach the same preregistered recovery protocol to every checkpoint.

The primary outcome is recovery advantage over a matched model that never learned the old task. Candidate predictors are new-task KL, immediate forgetting, new-task compression advantage, old-task code-length loss, fixed and freshly fitted probes, representational drift, update-subspace geometry, and local sharpness. The primary analysis tests out-of-sample prediction; the randomized information-load manipulation and full-versus-low-rank contrast test different parts of the mechanism. The result will establish either that KL is a sufficient practical signal for both forgetting and recovery, or that checkpoints at the same behavioral distance can differ because they wrote different information or took different geometric paths.

## 1. The problem

Suppose a model's score on an earlier task falls from 80% to 30% after fine-tuning. There are at least two operationally different explanations:

1. **Access loss:** useful internal structure remains, but the model no longer maps the task to the right behavior. A small alignment intervention may restore performance.
2. **Content loss:** task-specific information no longer provides a measurable learning advantage. Restoring performance requires reacquiring it from data.

The observed 50-point drop does not distinguish these cases. Neither does the fact that the final checkpoint moved far from the base model. The distinction matters in opposite directions for continual learning and unlearning: a practitioner wants ordinary forgetting to be cheap to reverse, while an unlearning claim is weak if a small intervention restores the target behavior.

What makes this problem scientifically exciting is that recovery may expose a hidden axis of model state. Two checkpoints can look equally damaged on task A, equally successful on task B, and equally far from the base under new-task KL, yet one may recover from a few task-format cues while the other may need the original facts and a substantial retraining budget. That pair is a natural experiment: the behavioral endpoint is held fixed while the latent condition that determines what can come back is allowed to vary. Finding a checkpoint-readable signal for that condition would turn forgetting from a scalar score into a diagnosis of what kind of intervention the model needs—and would reveal where a strong empirical law about *damage* stops being a theory of *retention*.

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

### 3.1 From catastrophic interference to recoverability

The original catastrophic-forgetting problem was formulated as **catastrophic interference**: after a connectionist network learns one mapping, gradient updates for a second mapping can abruptly destroy performance on the first ([McCloskey & Cohen, 1989](https://doi.org/10.1016/S0079-7421(08)60536-8)). This is an instance of the stability–plasticity dilemma. A learner must remain plastic enough to acquire task B while preserving structure that task A still needs. Because the task-B objective contains no term that values task A, ordinary gradient descent has no reason to protect that structure.

Classical continual-learning methods attacked different parts of this optimization problem. **Regularization** methods such as elastic weight consolidation penalize movement in parameters estimated to matter for previous tasks ([Kirkpatrick et al., 2017](https://doi.org/10.1073/pnas.1611835114)). **Rehearsal** methods interleave stored or generated task-A examples with task-B training. **Parameter-isolation and expansion** methods freeze, mask, or add components so new learning cannot freely reuse every old parameter. These families remain important engineering baselines, but their standard evaluation usually treats forgetting as one quantity: the drop in task-A performance after learning B.

That scalar view began to fracture when representation-level work showed that behavioral failure need not imply representational destruction. [Davari et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.html) found cases where a fresh linear classifier could still decode old-task labels after the deployed classifier failed. In language models, [Zheng et al. (2025)](https://arxiv.org/abs/2501.13453) showed that early fine-tuning can disrupt task alignment and produce severe but rapidly reversible behavioral forgetting. These results do not prove that all forgotten information survives; they show that terminal accuracy alone cannot tell us how much survives in a usable form.

Foundation-model post-training makes the distinction harder and more consequential. A single model supports many capabilities through shared representations and a shared output interface; there is no clean task-specific head whose failure can always be isolated. SFT, reinforcement learning, tool-use training, safety tuning, and domain adaptation can all improve a target behavior while moving unmeasured capabilities. [Shenfeld et al. (2025)](https://arxiv.org/abs/2509.04259) supply an unusually strong behavioral regularity—new-task forward KL predicts the immediate magnitude of forgetting—while unlearning research makes recovery an explicit adversarial test. The historical progression therefore leads directly to this proposal's pivot: **from asking only how much task-A performance fell to asking what evidence of task A remains and what budget is required to make it usable again.**

### 3.2 KL supplies the baseline predictor

Shenfeld et al. provide the empirical law this project tries to extend. Their result also supplies the first testbed, metric definition, hyperparameter-sweep strategy, oracle policy construction, and competing-predictor format. Their on-policy result is relevant because it explains how different training procedures populate different parts of the KL range; it is not evidence by itself about recovery.

Their distillation result should be interpreted narrowly. An SFT student trained on outputs from an RL teacher matched the teacher's measured learning–forgetting trade-off. This supports a final-policy account of *immediate forgetting on the measured distributions*. It does not establish that the two models are identical distributions everywhere, nor that their recovery dynamics must match.

### 3.3 Continual learning and unlearning supply the outcome

Zheng et al. show that severe behavioral forgetting can coexist with strong recovery after limited retraining, motivating access loss as a real phenomenon. Xu et al. formalize reversibility under a bounded relearning protocol and show that representation diagnostics can separate regimes within their unlearning setup. These papers justify measuring an entire recovery curve rather than treating the terminal accuracy drop as erasure.

[Davari et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.html) further show why behavioral and representational forgetting should be separated with linear probes. This proposal makes that distinction explicit: a probe fitted once at $\theta_0$ detects whether the old coordinate/readout remains stable, while a fresh probe fitted at each checkpoint detects whether task-A labels remain linearly decodable after the representation moves. A fresh probe is allowed to learn from A labels, so it measures usable information for that probe class—not zero-shot access by the deployed model and not literal bits stored in the weights.

Unlearning evidence is transferred cautiously. Unlearning objectives intentionally suppress a target behavior; ordinary fine-tuning optimizes a different task. A predictor that works in unlearning may fail in sequential fine-tuning. Testing that transfer is part of the contribution, not an assumption.

### 3.4 Compression supplies a measurement and a task-design principle

For a target dataset $D=\{(x_i,y_i)\}$, define the behavior-write quantity

$$
W(D;\theta_0,\theta)=\sum_i \left[-\log_2 p_{\theta_0}(y_i\mid x_i)+\log_2 p_{\theta}(y_i\mid x_i)\right].
$$

This is a change in conditional code length on specified targets. On uniformly random targets, where generalization is ruled out by construction, compression advantage is evidence of memorization. On natural text, $W$ mixes memorization, generalization, and changes in output convention. It is therefore a **behavioral code-length measure**, not a literal census of bits physically located in particular weights. Tan et al.'s excess statistic, which subtracts a size-matched held-out change, helps isolate training-specific information but does not remove every semantic or surface-form confound.

The distinction is load-bearing. The experiment separately measures:

- $B_{\text{new}}$: excess compression advantage on the new task, calibrated with random associations;
- $L_{\text{old}}$: increase in code length on old-task targets after fine-tuning;
- $C_{\text{recovery}}$: recovery behavior under a protocol that does not use either code-length quantity as its label.

### 3.5 What task design can and cannot guarantee

The input data are the cleanest available lever because the internal mechanism cannot be assigned directly.

| Fine-tuning content | External target information introduced? | Expected training-specific write | Role in the experiment |
|---|---:|---:|---|
| Random prompt–value associations or random strings | **Yes, by construction** | High if successfully learned | Positive control for deposition |
| Targets sampled or selected from the base model | No information outside the base/data-generation system | Often low, but not guaranteed to be zero | Sharpening/reweighting control |
| Deterministic format or style transformation | Little semantic information; the rule itself still has a description length | Near zero after held-out subtraction if the rule generalizes | Access/format control |

[Huang et al. (2025)](https://arxiv.org/abs/2412.01951) show why self-improvement cannot create information absent from the model and formalize self-improvement as sharpening. This constrains *epistemic novelty*. It does **not** imply that self-training makes zero parameter changes or that $W(D)$ must be exactly zero: a model can sharpen, re-encode, or memorize a sampled transcript without acquiring an externally supplied fact. Likewise, on-policy RL can only learn an exact random secret if it samples informative reward-bearing outputs; failure to do so is a coverage limitation, not a clean recovery comparison.

The primary causal contrast therefore uses SFT in every cell and changes only the prompt–target association. SFT versus RL is a secondary method comparison on tasks both methods can solve.

### 3.6 Low rank constrains the update; it does not name the mechanism

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

### 3.7 Scaling laws supply experimental discipline

The scaling-law literature matters here less because this project expects one universal power law and more because it demonstrates how easily an apparent law can be an artifact of the experimental budget. [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) and [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) reached different compute-optimal prescriptions after changing which data, model-size, and training-horizon regimes were measured. Later reanalyses and replications showed that warmup schedules, compute accounting, optimizer retuning, fitting procedures, and uncertainty estimates can materially change the fitted frontier ([Porian et al., 2024](https://arxiv.org/abs/2406.19146); [Besiroglu et al., 2024](https://arxiv.org/abs/2404.10102)). The lesson is not that empirical laws are fragile; it is that the law is only defined on a declared resource surface.

This proposal applies that lesson to forgetting and recovery. A single full-versus-LoRA operating point cannot distinguish update geometry from under-training, so learning rates and step budgets are tuned independently and comparisons are restricted to common support in task-B performance, KL, and immediate task-A damage. Writable capacity is reported at a declared exposure and optimization budget rather than treated as an architecture constant. Full parameter count, trainable parameter count, adapter placement, declared rank, and realized effective-update statistics are reported separately. Recovery is a response curve over examples, tokens, and optimizer steps—not a success/failure label at one arbitrary budget.

The predictive analysis follows the same discipline. Dense checkpoints from one run remain in one fold; entire mapping seeds and task templates are held out; later validation holds out a model scale. Functional forms are compared out of sample, and uncertainty and calibration are reported alongside $R^2$. The aim is therefore stronger than fitting a curve through a large sweep: it is to determine which relationship survives when the task, mapping, budget, and eventually scale change.

### 3.8 Why the answer matters

The project connects a mechanistic question to three concrete decisions:

1. **Remediation triage.** If a checkpoint mainly lost access, cue-only or small alignment datasets may recover task A without expensive replay. If its recovery advantage over a never-knew model has collapsed, the intervention must instead supply task-A content. A calibrated predictor can choose which audit to run first and estimate a plausible budget range.
2. **Unlearning verification.** Suppressed outputs are not evidence that target information is difficult to recover. A procedure that only changes access may satisfy a static benchmark while failing the operational deletion goal. Recovery curves, content-sensitive code length, and never-knew controls provide a stricter audit, while the proposal remains careful not to equate finite-budget resistance with proof that information is absent from every weight.
3. **Post-training and continual-learning design.** KL-conservative objectives, replay, freezing, full fine-tuning, and parameter-efficient updates can be compared not only by task-B learning and immediate retention but by the future cost of restoring task A while preserving B. This matters when original data may later be unavailable, when a model will undergo many sequential updates, or when a consolidation pipeline is about to discard an adapter or checkpoint that contains the cheapest route back.

The deployment limit is equally important. Checkpoint statistics cannot reconstruct task-A facts that are genuinely unavailable, and a no-old-data predictor cannot certify post-intervention success without some definition or evaluation of task A. The realistic product is a triage policy that recommends a treatment class, budget range, confidence, or explicit abstention—not an oracle that recovers unspecified knowledge from weights alone.

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

The design separately identifies **novelty** from **conflict**. Permuting an association attached to a familiar key can both introduce information and override probability mass that $	heta_0$ assigns to the original association. A fresh-key cell, verified to have no systematic base preference over candidate values, supplies novel associations without that contradiction.

### RQ4 — Rank and update geometry

Does rank-8 LoRA change recovery relative to full fine-tuning after matching task-B performance, new-task KL, and immediate task-A forgetting?

**H4a.** At matched task-B performance, rank-8 LoRA will usually show less immediate task-A forgetting than full fine-tuning. This tests whether a published learning–forgetting pattern transfers to the present setting and is not yet the recovery contribution.

**H4b.** After additionally matching KL and immediate forgetting, update parameterization will retain incremental predictive value for the recovery curve if endpoint behavioral distance is not sufficient. The directional mechanism is deliberately tested rather than assumed: an access/reorientation result would combine a larger fixed-versus-fresh probe gap with preserved old-task code length and positive recovery advantage; a rewrite result would combine fresh-probe loss, old-task code-length loss, and recovery approaching the never-knew baseline.

**H4c — angle-conditioned rank invariance.** Steele's recent preprint predicts that LoRA rank has little effect on immediate forgetting when the principal angle between task-gradient subspaces is high, with stronger rank effects when task subspaces are similar. The direct extension tested here is: **does rank stop mattering for what comes back, too, when task-subspace angles are high?** Convergence of rank-conditioned recovery curves in the high-angle regime would extend the proposed law from forgetting to recoverability; divergence would establish its boundary.

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

This manipulation does **not** change only novelty. When a prompt already elicits probability mass on its original value, permutation also creates conflict with a base prior: the model must suppress an existing association while learning the reassigned one. Initial loss and base target probabilities measure this conflict but do not remove it.

Add a paired **fresh-key cell** to isolate novel information without override. Generate synthetic keys outside the task-A and base-derived key templates, pair them with the same target-string multiset, and filter or stratify them so $	heta_0$ has no systematic preference among candidate target assignments. Match target lengths, token marginals, dataset size, exposures, and optimization exactly to the permuted-key cell. The contrast is then:

- fresh key + novel value: novelty with minimal measured conflict;
- familiar key + permuted value: novelty plus override pressure.

Fully random token strings remain a capacity-calibration condition, not the sole novelty treatment, because natural versus random strings would otherwise confound information load with surface form and initial likelihood.

Add a deterministic formatting cell as a low-information positive-learning control. Measure initial loss and gradient statistics in every cell rather than assuming the construction matches difficulty perfectly.

#### Capacity pressure

For each update parameterization, and for every LoRA placement/rank ablation, estimate writable capacity using the random-string protocol of Morris et al. and Tan et al. Define

$$
\rho = \frac{\text{entropy of novel task-B associations}}{\text{measured writable capacity at the declared training budget}}.
$$

Run at several $\rho$ bands below and around one. Capacity claims require this axis: writing novel facts into a largely empty parameterization does not imply that old facts must be displaced.

#### Creating overlap

Sweep learning rate and training steps independently within every $(\lambda,\rho,\text{parameterization},\text{conflict condition})$ cell. Retain a common-support region in which cells overlap on:

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
| Base-association conflict | Initial target loss and base probability margin between assigned and original values | Separates fresh-key novelty from familiar-key override pressure; it is a measured covariate, not a substitute for the randomized fresh-key cell. |
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

Recovery is counted only while task B remains within a preregistered tolerance of its post-fine-tuning score. For recovery checkpoint $\theta_{A\leftarrow B}(b)$ at budget $b$, admissibility requires

$$
S_B\!\left(\theta_{A\leftarrow B}(b)\right)
\geq
S_B(\theta_B)-\epsilon_B,
$$

where $\epsilon_B$ is set from task-B evaluation noise before recovery and accompanied by a sensitivity analysis. Every recovery metric is computed on this admissible portion of the curve; if the threshold is never reached without violating the B floor, the recovery budget is censored. The full joint $(S_A,S_B)$ Pareto frontier is reported so that “recovering A by forgetting B back” cannot count as successful recovery.

For each arm, evaluate a budget ladder in examples, tokens, and optimizer steps. Use the best result from a small preregistered recovery-learning-rate grid so that a checkpoint is not labeled resistant merely because one learning rate was unsuitable.

Primary recovery outcomes are:

- **B-constrained recovery AULC:** area under the admissible task-A recovery curve on a log-budget axis;
- **B-constrained recovery advantage:** admissible AULC from $\theta_B$ minus admissible AULC from the matched never-knew-A control;
- **Joint recovery frontier:** the Pareto frontier of task-A restoration and task-B retention over budgets and recovery learning rates;
- **Recovery ceiling:** best admissible task-A score within the maximum budget;
- **$C_{90}$:** minimum admissible budget restoring 90% of the lost score, reported as a secondary, censored outcome with 80% and 95% sensitivity analyses.

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
| Geometry | High- vs low-A/B gradient-subspace-angle task pairs | Does rank stop mattering for forgetting—and for what comes back—when task angles are high? |
| Temporal dissociation | Dense early task-B checkpoints | Does task-A damage appear before old-task code-length loss, as an access-loss account predicts? |
| Bias/scaling | Frozen vs trainable bias; fixed $\alpha/r$ | Are implementation details masquerading as rank effects? |

These ablations are ordered to control scope. Week 1 runs only the primary full-versus-$r=8$ comparison, with all linear layers targeted and separate learning-rate sweeps. Phase 1 adds $r\in\{2,8,32\}$ if the primary contrast has common support and a detectable effect. Placement and matched-dimensional random-subspace controls are LLM-stage mechanism checks.

The **temporal dissociation** is a secondary, step-resolved analysis that can be completed before the recovery arms. Log task-A performance, old-task code length, fixed/fresh probes, and new-task KL at logarithmically spaced early B-training steps in addition to the existing fractional checkpoints. Spurious-forgetting theory predicts a sharp early task-A drop while old-task code-length loss remains small; a rewrite account predicts behavioral damage and code-length loss moving together. A clean lag is strong evidence for early access disruption, although later content loss may still accumulate. This analysis requires extra checkpoint evaluation but no recovery training.

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

Estimate the randomized effects of $\lambda$, $\rho$, conflict condition, parameterization, and their preregistered interactions within the common-support region, controlling for task performance, KL, and immediate forgetting. The fresh-key versus familiar-permuted contrast identifies override pressure beyond novelty. The information-load × capacity-pressure interaction is the confirmatory capacity test. The rank-8/full coefficient and parameterization × task-subspace-angle interaction test whether update geometry adds information beyond the behavioral endpoint. A preregistered high-angle equivalence test asks whether recovery differences across LoRA ranks fall inside a practical-equivalence band. Mediation through measured $B_{\text{new}}$, $L_{\text{old}}$, or probes is exploratory because those quantities are post-treatment variables.

#### Falsification checks

- Held-out random associations must have near-zero training-specific compression advantage.
- Fresh keys must show no systematic base preference among candidate values; otherwise the zero-conflict interpretation fails and conflict remains a measured covariate only.
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
| Week 1 | Run the four-cell objective × parameterization pilot: SFT-1 and 1–0 REINFORCE, each with full fine-tuning and all-layer LoRA $r=8$; verify exact zero-step equivalence, trainable-parameter accounting, effective-weight metrics, bits, CKA, fixed/fresh probes, and logarithmic early-step logging. | P0 harness and LoRA correctness |
| Weeks 2–3 | Full Phase-0 sweep with independently tuned full/LoRA grids and at least three seeds; estimate common support and preregister the recovery protocol. | K1, K6 |
| Weeks 4–5 | Build paired A/B synthetic task, never-knew control, and random-association calibration. | K5 |
| Weeks 6–7 | Information-load × capacity-pressure sweep and recovery arms. | K2–K4 |
| Weeks 8–9 | Grouped predictive analysis, ablations, and held-out-task validation. | Toy-stage claim |
| Thereafter | Reduced-density LLM validation and, separately, unlearning transfer. | External validity |
| Phase 2 / future | Fit and validate the no-old-data treatment-and-budget triage policy across held-out task families. | Deployable decision support |

## References

- Aghajanyan, A., Gupta, S., & Zettlemoyer, L. (2021). [*Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*](https://arxiv.org/abs/2012.13255).
- Besiroglu, T., et al. (2024). [*Chinchilla Scaling: A Replication Attempt*](https://arxiv.org/abs/2404.10102).
- Biderman, D., et al. (2024). [*LoRA Learns Less and Forgets Less*](https://arxiv.org/abs/2405.09673).
- Davari, M., et al. (2022). [*Probing Representation Forgetting in Supervised and Unsupervised Continual Learning*](https://openaccess.thecvf.com/content/CVPR2022/html/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.html). CVPR 2022.
- Fan, C., et al. (2025). [*Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond*](https://arxiv.org/abs/2502.05374).
- Hoffmann, J., et al. (2022). [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556).
- Hu, E. J., et al. (2021). [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
- Huang, A., et al. (2025). [*Self-Improvement in Language Models: The Sharpening Mechanism*](https://arxiv.org/abs/2412.01951). ICLR 2025.
- Jin, H., et al. (2026). [*Rotation-Preserving Supervised Fine-Tuning*](https://arxiv.org/abs/2605.10973). Preprint.
- Kaplan, J., et al. (2020). [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361).
- Kirkpatrick, J., et al. (2017). [*Overcoming Catastrophic Forgetting in Neural Networks*](https://doi.org/10.1073/pnas.1611835114). PNAS.
- McCloskey, M., & Cohen, N. J. (1989). [*Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem*](https://doi.org/10.1016/S0079-7421(08)60536-8).
- Morris, J. X., et al. (2025). [*How Much Do Language Models Memorize?*](https://arxiv.org/abs/2505.24832).
- OpenUnlearning / Dorna, V., et al. (2025). [*Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics*](https://arxiv.org/abs/2506.12618).
- Porian, T., et al. (2024). [*Resolving Discrepancies in Compute-Optimal Scaling of Language Models*](https://arxiv.org/abs/2406.19146). NeurIPS 2024.
- Rybak, P., et al. (2026). [*REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop*](https://arxiv.org/abs/2602.06248).
- Shenfeld, I., Pari, J., & Agrawal, P. (2025). [*RL's Razor: Why Online Reinforcement Learning Forgets Less*](https://arxiv.org/abs/2509.04259).
- Steele, B. (2026). [*Subspace Geometry Governs Catastrophic Forgetting in Low-Rank Adaptation*](https://arxiv.org/abs/2603.02224). Preprint.
- Tan, K., Du, H., & Feng, Y. (2026). [*How Many Bits Can an Adapter Write? Measuring the Capacity and Memorization of Parameter-Efficient Fine-Tuning*](https://arxiv.org/abs/2607.21351).
- Xu, X., et al. (2025). [*Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs*](https://arxiv.org/abs/2505.16831).
- Zheng, J., Cai, X., Qiu, S., & Ma, Q. (2025). [*Spurious Forgetting in Continual Learning of Language Models*](https://arxiv.org/abs/2501.13453). ICLR 2025.
