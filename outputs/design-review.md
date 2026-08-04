# Design review: *Distance Is Not Damage*

## Bottom line

The original proposal had a strong research question but did **not yet cleanly test its mechanistic
hypothesis**. It could test whether several checkpoint statistics correlate with a bounded recovery
protocol. It could not support the stronger conclusion that writing new bits causes irreversible
forgetting, because task type, information load, learnability, and capacity pressure were confounded.

The revised design fixes the main identification problems. If carried out as written, it can answer:

1. whether new-task KL predicts recovery after ordinary fine-tuning;
2. whether a controlled increase in externally supplied information changes recovery at matched KL
   and immediate forgetting; and
3. whether that effect becomes stronger near a measured capacity boundary.

It still cannot prove literal erasure or universal irreversibility. Those are not finite-experiment
observables. Its defensible outcome is *recovery advantage under a declared budget, relative to a
matched model that never learned the old task*.

## What was already strong

- The distinction between forgetting magnitude and recoverability is important and operational.
- RL's Razor provides a natural baseline law, testbed, and predictor-ablation format.
- Reversibility work in continual learning and unlearning supplies concrete recovery protocols.
- The idea of manipulating task content, rather than pretending to assign an internal mechanism,
  is the right instinct.
- Surface-form sensitivity, numeric precision, seed variance, and kill criteria were treated as real
  design constraints rather than afterthoughts.

## The major problems in the original draft

### 1. The contribution was buried under several adjacent claims

The draft alternated among four projects: extending the KL law, explaining the SFT–RL gap, testing a
capacity mechanism, and improving unlearning evaluation. Those projects are related, but they are
not one contribution.

The revised story is hierarchical:

> KL predicts immediate loss. Recovery is a different outcome. First test whether KL transfers to
> that outcome; then ask whether controlled information load explains residual recovery variation.

Unlearning now motivates the outcome and supplies comparators; it is not presented as the primary
domain.

### 2. The literature gap was overstated

The claim that “nothing predicts which reversibility regime a model is in” was too strong. Xu et al.
report that representation drift predicts recovery under their fixed unlearning protocols.
OpenUnlearning evaluates metric faithfulness and robustness, while REBEL studies adversarial prompt
recovery; neither supports the original specific claim that relearning-curve shapes fail as general
predictors.

The defensible gap is narrower and cleaner: no cited work tests the new-task KL law against recovery
after ordinary fine-tuning, and no cited work performs a matched-KL causal information-load test.

### 3. “Bits written into weights” conflated three quantities

The original draft treated

$$
W(D)=\sum_{(x,y)\in D}\left[-\log_2p_{\theta_0}(y\mid x)+\log_2p_{\theta}(y\mid x)\right]
$$

as a direct readout of information physically stored in weights. On ordinary text it is a behavioral
code-length change on selected targets. It can reflect generalization, memorization, sharpening, or
format change. Tan et al. explicitly distinguish parameter codelength, random-data memorization, and
behavior-write bits.

The revision separately names new-task deposition, old-task code-length loss, and the recovery
outcome. Random associations calibrate the first quantity; paraphrase/content-token variants probe
the second quantity's format sensitivity.

### 4. Random facts versus self-generated outputs was not an isolated manipulation

Random strings, model-generated answers, and style targets differ in initial likelihood, semantics,
gradient scale, target entropy, output length, and attainable performance. Matching only KL does not
remove those confounds.

The revised primary treatment permutes target assignments across the same prompts. This preserves
the exact target strings and marginal token distribution while randomizing how much prompt–target
association the base cannot know. Fully random strings remain a calibration condition.

### 5. The capacity mechanism had no capacity-pressure intervention

Writing a novel fact does not imply displacement when the trainable parameterization has abundant
unused capacity. A capacity account predicts an interaction: novel information should make recovery
harder especially as information load approaches writable capacity.

The revision calibrates writable capacity for each adapter configuration and varies the load ratio
$\rho$. Without an information-load × capacity-pressure effect, the proposal will not claim capacity
displacement.

### 6. Raw relearning speed did not distinguish retention from ordinary learnability

A model that truly lost task A may still relearn it quickly because the task is easy or shares useful
features with task B. The original design lacked a “never knew A” recovery curve.

The revision preserves a pre-A checkpoint, subjects it to the same B procedure, and compares recovery
from the forgot-A checkpoint with recovery from this matched never-knew-A control. The difference in
recovery AULC is the primary retention signal.

### 7. “Irreversible” was stronger than the measurement

Failure to recover at 500 samples and three epochs is failure under that budget, not proof that no
intervention can recover the capability. The revised terminology is *budget-resistant*. It reports
the full data/compute ladder, ceiling, censored threshold outcome, and protocol sensitivity.

### 8. RL on unguessable random secrets was not a fair matched-task cell

Policy-gradient training receives useful signal about an exact random code only after sampling it.
If the base assigns negligible probability, RL cannot reach matched performance. That null gradient
is a coverage result, not evidence about recovery conditional on learning.

SFT is therefore used in every primary information-load cell. SFT versus on-policy methods is a
secondary comparison restricted to outputs both methods can learn.

### 9. The regression plan mixed prediction and mechanism

A large sweep followed by regression can identify a predictor, but it cannot by itself identify a
mechanism. Dense checkpoints also create pseudo-replication if checkpoints from one run appear in
both train and test folds.

The revision separates:

- grouped held-out prediction across entire mappings/tasks; and
- causal estimation of randomized information load and capacity pressure inside a common-support
  region.

Post-treatment “bits” mediation is labeled exploratory.

### 10. ParityMNIST was being asked to do too much

ParityMNIST is well suited to reproducing the KL law because many output distributions are equally
correct. Its ten-label output space is poorly suited to a high-dynamic-range information-capacity
test. It is now Phase 0 only. The causal bits experiment moves to a decoder/LoRA setting with an
explicit capacity calibration.

### 11. LoRA rank was being mistaken for a mechanism label

The claim “LoRA confines memorization/overwrite while full SFT rotates representations” does not
follow from the literature. SFT is an objective; LoRA is an update parameterization, so they must be
crossed rather than contrasted. A rank-$r$ adapter constrains each targeted matrix update to rank at
most $r$, but low rank is not necessarily an orthogonal rotation or pure memorization, and a
high-rank full update is not proof of erasure.

The revised design treats full fine-tuning versus all-layer LoRA rank 8 as a primary practical
contrast from Week 1. It tunes each parameterization separately and compares recovery only in common
support on task-B performance, KL, and immediate task-A damage. Rank $\{2,8,32\}$, placement,
matched-dimensional random subspaces, and task-gradient principal angles are staged mechanism
ablations. Adapter-off performance is only a sanity check because it restores A by deleting B.

## Does the revised plan test the hypotheses?

| Hypothesis | Test | What would falsify it? |
|---|---|---|
| H1: KL predicts immediate forgetting | Multi-method ParityMNIST sweep; held-out fit across seeds | Unstable or weak KL relationship after a faithful reproduction |
| H2: KL predicts recovery | Same recovery assay on every checkpoint; grouped held-out comparison with/without KL | No out-of-sample gain over immediate forgetting alone |
| H3: information load affects recovery through capacity pressure | Randomized association fraction × calibrated load ratio, matched on KL/damage/performance | No interaction in the common-support region |
| H4: update parameterization adds recovery information beyond the endpoint | Full fine-tuning vs rank-8 LoRA, matched on B performance, KL, and immediate A forgetting | No held-out recovery difference or incremental prediction after matching |
| Access-loss signature | Cue-only recovery containing no A facts; code-length and representation diagnostics | Cue intervention performs like never-knew control |
| Rewrite/capacity signature | Recovery advantage over never-knew control, old-task code-length loss, load × pressure effect | Fast recovery advantage persists at high load/pressure |
| Sharpness predictor | Preregistered normalized curvature measures added to held-out model | No stable incremental predictive value |

Yes: the revised plan tests these operational hypotheses. No finite version of the plan can prove
that a capability is metaphysically “gone,” and a significant predictor should not be described as
the unique internal mechanism.

## Recommended claim discipline

Use:

- “predicts recovery under the preregistered budget”;
- “supports a capacity-constrained deposition account”;
- “is consistent with access loss”;
- “adds held-out predictive signal beyond KL and immediate forgetting.”

Avoid:

- “reads the bits in the weights” on natural data;
- “proves irreversible erasure”;
- “self-generated training writes zero bits”;
- “the final distribution is identical” based on matched benchmark behavior;
- “path-based accounts die” after one distillation comparison.
