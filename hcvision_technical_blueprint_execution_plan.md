# HCVISION R&D Project: Technical Blueprint & Execution Plan

**Course:** R&D Experience (HCVISION) — Academic Year 2026–2027

**Dataset:** EPIC-KITCHENS-100 (Video-Only Modality, $640 \times 360$, 700 videos, ≈49 GiB on VSC; the course page's \~17 GB figure is too low)

**Deadlines:** Mid-Term Presentation: 13/11/2026 | Final Report, Code, Poster & Defense: 11/12/2026

> **Revision 2026-11-05:** corrected the dataset size, the V-JEPA 2.1 checkpoint resolution (§4), the $640 \times 360$ preprocessing description (§4.1), the feature-bank storage estimate and predictor caching (§4.3), the leakage-rule conflict of Option B and the gate-interpretability claim (§5.3), the submission format (§6.2), and Week 1 tasks that already exist in the repository (§7). Choices that still need a team decision are marked **Decision needed**.

## 1. Grading Strategy & Core Research Thesis

### 1.1 Why Design Choices Matter More Than Brute Force

Our grade is split across five deliverables:

* **Presentations & Defense (45%):** Mid-term (15%) + Final presentation and defense (30%).

* **Conference-Format Report & Poster (30%):** Final paper (20%) + Poster (10%).

* **Solution Quality (25%):** Leaderboard results, code organization, reproducibility, documentation, and implementation quality.

To score $17+/20$, our project must read like a cohesive CVPR/ICLR workshop paper: every module must solve a specific flaw in existing papers, be backed by clean mathematical normalization, and be validated across all benchmark sub-metrics (Overall, Unseen Participants, and Tail Classes).

### 1.2 Our Core Scientific Thesis

We combine and extend three 2026 state-of-the-art paradigms on EPIC-KITCHENS-100:

1. **V-JEPA 2.1 Dense Spatiotemporal Representations** (Mur-Labadia et al., 2026; arXiv:2603.14482) and **JFAA Attentive Probing** (Chu et al., 2026; arXiv:2605.20904, 1st place CVPR 2026 EgoVis Challenge).

2. **Action-Guided Attention (AGA)** (Tai et al., 2026; arXiv:2603.01743, ICLR 2026), which uses past action probability predictions as Queries and Keys to retrieve long-term visual memory.

3. **Metric-Aware Dual Logit Calibration** (extending Menon et al., 2020), aligning marginal verb/noun logits with the opposing evaluation metrics of Task 1 (AR) and Task 2 (STA).

#### What We Make Unique (Our 3 Contributions):

1. **Factorized Spatiotemporal-Motion Attentive Probing (Both AR & STA):** Instead of global average pooling (which destroys spatial detail and overfits to background kitchens on unseen participants) or single-stream probing, we extract a compact Temporal-Tubelet + Spatial-Grid representation from V-JEPA 2.1 and use disentangled Verb Queries (attending to temporal progression) and Noun Queries (attending to spatial active-object tokens).

2. **Decoupled Verb-AGA & Noun-AGA with Normalized 3-Way Gating (STA):** Existing AGA concatenates predictions into a single stream. However, in kitchen workflows, nouns persist locally ($\text{take plate} \rightarrow \text{wash plate} \rightarrow \text{put plate}$) while verbs follow state transitions ($\text{open} \rightarrow \text{take} \rightarrow \text{close}$). We split AGA into coupled Verb and Noun memory streams and fuse Long-Term Memory ($h_t$), Present Visual Evidence ($e_t$), and Anticipated Latent Tokens ($f_t$) via a LayerNorm-Calibrated 3-Way Gate.

3. **Task- and Metric-Specific Logit Adjustment + Procedural Cold-Start Prior:** Because Codabench only accepts marginal `verb_output` ($97\text{-D}$) and `noun_output` ($300\text{-D}$) score vectors, and evaluates AR on Action Top-1 Accuracy (head-biased) vs. STA on Action Class-Mean Top-5 Recall / MT5R (tail-balanced), we derive separate post-hoc temperature/prior adjustments $(\tau_v, \tau_n)$ for AR and STA, plus a Markov transition prior to handle hidden-test events starting before $1.0\text{s}$.

## 2. Task Rules, Metrics & Leakage Prevention

### 2.1 Task 1: Action Recognition (AR)

* **Input:** Trimmed video segment $[t_s, t_e]$ containing an action.

* **Output:** Marginal score vectors $\text{verb\_output} \in \mathbb{R}^{97}$ and $\text{noun\_output} \in \mathbb{R}^{300}$.

* **Primary Metric:** Action Top-1 Accuracy (plus Verb/Noun Top-1/Top-5, Unseen Participants, and Tail Classes).

### 2.2 Task 2: Short-Term Action Anticipation (STA)

* **Input:** Video observed strictly up to $t_{\text{obs}} = t_s - 1.0\text{s}$ (`sta_observation_end_seconds`).

* **Strict Anti-Leakage Rule:** Zero frames from $(t_s - 1.0\text{s}, t_e]$ may be read during training or testing, including during feature extraction.

* **Cold-Start Edge Case (**$t_s < 1.0\text{s}$**):** When an action starts before $1.0\text{s}$ in the video, $t_{\text{obs}} < 0$, so no valid video frames exist. We must output a prior-based STA prediction without reading future frames.

* **Primary Metric:** Action Class-Mean Top-5 Recall (MT5R), which averages Top-5 recall equally across all classes present in the evaluated split.

## 3. Data Engineering: The Hybrid Validation Split

> *"Use the provided training annotations only for development. Each team chooses and documents its own train/validation split within the provided train set."*

### 3.1 Why Naive Splits Fail

* **Random segment split:** Adjacent clips from the same video (`P01_01`) share identical lighting and kitchen backgrounds, causing severe data leakage and inflated validation accuracy.

* **Holding out only whole participants:** If 100% of our validation set consists of held-out participants, we can only measure Unseen Participant performance and are blind to Seen Participant performance and rare tail classes.

### 3.2 Our Hybrid Seen + Unseen Validation Split (\~12–15% of Train Data)

1. **Unseen-Participant Val Subset (\~6% of clips):** Write a 10-line exploratory script on `train.csv` to inspect clip counts per participant, and hold out 2–3 medium-sized participants completely (zero videos in train). This directly mirrors the leaderboard's Unseen Participants metric.

2. **Seen-Participant Val Subset (\~8% of clips):** For every remaining participant, sort their videos chronologically by video ID (`PXX_01`, `PXX_02`, ...) and place the last 10–15% of complete videos into validation.

3. **Coverage Verification Script:** Verify that at least 95%+ of the training verb and noun classes appear in the combined validation set, and flag the tail classes (the smallest classes accounting for the bottom 20% of training instances) so we can compute Overall, Unseen-Participant, Seen-Participant, and Tail-Class metrics locally.

## 4. Feature Extraction & Temporal Alignment on VSC

Decoding ≈49 GiB of compressed H.264 video on every training epoch will bottleneck VSC GPUs on CPU/disk I/O. We extract features once using a frozen V-JEPA backbone ([github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)) and save compact fp16 tensors.

**Selected 2026-11-06:** V-JEPA 2.1 distilled ViT-L/384, checkpoint
`vjepa2_1_vitl_dist_vitG_384.pt`. The input is 384 x 384, 16 frames at 8 fps
(2 seconds). V-JEPA 2 ViT-L/256 is retained only as a predictor-fidelity
reference; it is not the production feature-bank backbone.

### 4.1 Spatial Preprocessing (640 x 360 to 384 x 384)

Use the selected model's official evaluation transform for the extraction baseline.
Resizing the short edge to 384 gives approximately 683 x 384, then a 384 x 384
center crop retains about 56% of the width. Alternatively, full-frame resizing
to a square preserves the field of view but distorts the aspect ratio.
The current Member 4 feasibility script uses that full-frame resize with ImageNet
normalization. Its timing/fidelity measurements describe this diagnostic transform,
not an official evaluation result. Feature-bank meta.json must identify the exact
resize/crop and normalization used; do not mix caches from different transforms.

### 4.2 Exact Temporal Alignment (Preventing Leakage & Context Loss)

* **Why a fixed** $1.0\text{s}$ **integer grid fails:** If an action starts at $t_s = 14.73\text{s}$, then $t_{\text{obs}} = 13.73\text{s}$. Using a fixed integer grid ($13.0\text{s}$) throws away $0.73\text{s}$ of the most informative pre-contact motion right before the gap! Rounding up ($14.0\text{s}$) is illegal leakage.

* **Event-Anchored STA Extraction:**

  * For each event $i$ with cutoff $t_{\text{obs}}^{(i)} = \text{sta\_observation\_end\_seconds}$, define $S = 16$ historical anchor steps spaced by $\Delta = 1.0\text{s}$:
    

    $$
    \tau_k^{(i)} = t_{\text{obs}}^{(i)} - (S - k) \cdot 1.0\text{s}, \quad k \in \{1, 2, \dots, S\}
    $$

  * Notice that the final step $k = S$ ends exactly at $\tau_S^{(i)} = t_{\text{obs}}^{(i)}$.

  * To save compute on dense videos, extract a dense strided feature bank across each untrimmed video (stride $0.25$–$1.0\text{s}$; see the cost note in §4.3) (each clip covering a $2.0\text{s}$ window $[t - 2.0\text{s}, t]$ of 16 frames at 8 fps), **PLUS** an exact clip ending at $t_{\text{obs}}^{(i)}$ for the final step $k=S$. For earlier memory steps $k < S$, snap $\tau_k^{(i)}$ down to the nearest bank entry $\le \tau_k^{(i)}$. This guarantees zero future leakage, exact alignment at $t_{\text{obs}}^{(i)}$, and fast extraction.

* **Segment-Aligned AR Extraction:**

  * Because many AR segments $[t_s, t_e]$ are shorter than $2.0\text{s}$, sample 16 frames uniformly inside $[t_s, t_e]$ (using dense multi-crop sampling of 2 clips for segments longer than $4.0\text{s}$).

### 4.3 Compact Token Storage Format (Solving the Pooling vs. Probe Dilemma)

For a 16-frame clip at $256 \times 256$ (V-JEPA 2 ViT-L/256), the encoder outputs $T' \times H' \times W' = 8 \times 16 \times 16 = 2048$ tokens of dimension $D = 1024$; at $384 \times 384$ (V-JEPA 2.1 ViT-L/384) it outputs $8 \times 24 \times 24 = 4608$ tokens. Saving all 2048 tokens takes too much disk space, but saving a single mean-pooled vector ($1 \times D$) breaks the JFAA Attentive Probe.

#### Our Compact Token Representation ($24 \times D$ per clip in fp16):

1. **Temporal Motion Tubelets (**$E_{\text{temp}} \in \mathbb{R}^{8 \times D}$**):** Spatial average pool of each of the 8 temporal slices. Preserves frame-to-frame temporal dynamics for Verb recognition/anticipation.

2. **Active Spatial Grid (**$E_{\text{spat}} \in \mathbb{R}^{16 \times D}$**):** $4 \times 4$ adaptive spatial average pool over the last 2 temporal slices (the final $0.5\text{s}$ of the clip). Preserves spatial layout of hands and countertop objects for Noun recognition/anticipation.

* **Disk Footprint:** $24 \times 1024 \times 2\text{ bytes} \approx 49\text{ KB}$ per clip. EK-100 has ≈100 h of video (≈360,000 s), so a $0.25\text{s}$ bank is ≈1.44 M clips ≈ 70 GB, plus the per-event AR/STA clips; a $0.5\text{s}$ bank is ≈35 GB and a $1.0\text{s}$ bank ≈18 GB. The number of clips is also the number of backbone forward passes, which dominates extraction time. **Decision needed:** a $0.5\text{s}$ stride keeps the snapping error of the $1.0\text{s}$-spaced memory steps at $\le 0.5\text{s}$ for half the cost. Store the bank as per-video shards and load them on demand instead of holding it all in RAM.

* **Cache the predictor output in the same pass:** Module 3 Option A runs the frozen V-JEPA predictor on the *full* encoder token sequence plus mask tokens, so it cannot be run later from the 24 compressed tokens. Run the predictor during extraction for every per-event STA clip and store its future tokens in the same compressed format; otherwise all STA features must be re-extracted in Week 5.

## 5. Model Architecture (AR & STA)

### 5.1 Module 1: Factorized Spatiotemporal Attentive Probe (Used in Both AR & STA)

Given the 24 cached tokens $Z_t = [E_{\text{temp}}; E_{\text{spat}}] \in \mathbb{R}^{24 \times D}$:

* Instead of a single pooled vector, we introduce learnable query vectors: Verb Queries $q_v \in \mathbb{R}^{M_v \times D}$ and Noun Queries $q_n \in \mathbb{R}^{M_n \times D}$ (with $M_v = M_n = 4$ multi-queries pooled into $D$).

* **Factorized Routing (Ablation vs. Full Attention):**

  * In Full Attention (JFAA baseline), both $q_v$ and $q_n$ cross-attend over all 24 tokens $Z_t$.

  * In Factorized Spatiotemporal Routing (Ours), $q_v$ cross-attends primarily over the 8 temporal motion tokens $E_{\text{temp}}$ (plus global context), while $q_n$ cross-attends over the 16 spatial grid tokens $E_{\text{spat}}$.

* This outputs disentangled present embeddings:
  

  $$
  e_t^v \in \mathbb{R}^{D}, \quad e_t^n \in \mathbb{R}^{D}
  $$

* **For Task 1 (AR):** $e_{\text{seg}}^v$ and $e_{\text{seg}}^n$ pass through a cross-interaction MLP (so verbs know which object is held and vice versa) directly to the 97-way Verb and 300-way Noun classifiers.

### 5.2 Module 2: Decoupled Verb-AGA & Noun-AGA Memory (STA Only)

Following AGA (Tai et al., 2026; arXiv:2603.01743), over the $S = 16$ historical steps leading up to $t_{\text{obs}}$, each step $k \in \{1, \dots, S\}$ produces auxiliary verb and noun probability distributions:

$$
\hat{p}_k^v = \text{softmax}(W_v^{\text{aux}} e_k^v) \in \Delta^{96}, \quad \hat{p}_k^n = \text{softmax}(W_n^{\text{aux}} e_k^n) \in \Delta^{299}
$$

#### Exact Specification of $\hat{p}_k$ (No Ambiguity):

1. **What does step** $k$ **predict?** At every step $\tau_k$, the auxiliary head is supervised with the label of the next upcoming action $(v_*, n_*)$ if an action begins within $[\tau_k + 0.5\text{s}, \tau_k + 2.5\text{s}]$; steps with no upcoming action in that window are masked out of the auxiliary loss (or assigned to an auxiliary background slot) while still outputting a valid probability vector $\hat{p}_k$ into the FIFO memory queue.

2. **Teacher Forcing vs. Self-Predictions:** Keys and Queries always use `stop_gradient` ($\text{detach()}$) model predictions $\hat{p}_k$ during both training and inference—never ground-truth one-hot labels—so there is zero train/test distribution mismatch.

3. **Decoupled Verb and Noun Memory Retrieval:**

   * Compute Exponential Moving Average (EMA) queries ($\alpha = 0.8$):
     

     $$
     q_t^v = \alpha \hat{p}_t^v + (1 - \alpha) q_{t-1}^v, \quad q_t^n = \alpha \hat{p}_t^n + (1 - \alpha) q_{t-1}^n
     $$

   * Project queries and keys into $d_k = 128$ and retrieve separate Verb History $h_t^v$ and Noun History $h_t^n$:
     

     $$
     h_t^v = \text{Attention}\big(Q = q_t^v W_Q^v,\; K = \hat{P}_{t-S:t-1}^v W_K^v,\; V = E_{t-S:t-1}^v W_V^v\big)
     $$

     $$
     h_t^n = \text{Attention}\big(Q = q_t^n W_Q^n,\; K = \hat{P}_{t-S:t-1}^n W_K^n,\; V = E_{t-S:t-1}^n W_V^n\big)
     $$

### 5.3 Module 3: Near-Future Latent Predictor ($f_t$) & Normalized 3-Way Gate

To bridge the $1.0\text{s}$ anticipation gap $[t_{\text{obs}}, t_s]$, we compare two legal future-latent estimators:

* **Option A (Frozen V-JEPA 2.1 Predictor, as in JFAA arXiv:2605.20904):** Feed the observed clip tokens at $t_{\text{obs}}$ into the frozen V-JEPA 2.1 predictor with future mask tokens to obtain predicted tokens $\hat{Z}_{t+1}$, then apply the Factorized Attentive Probe to get $(f_t^v, f_t^n)$.

* **Option B (Learned Causal Latent Bridge — Ablation, pending a rule check):** A lightweight 2-layer causal Transformer trained to regress the stop-gradient V-JEPA encoding of the target action segment $e_{\text{AR}}$ from the observed sequence $e_{t-S:t}$ using a cosine similarity loss during training (requiring zero future frames at inference). **Rule check needed:** §2.2 also forbids reading frames from $(t_s - 1.0\text{s}, t_e]$ during training, and Option B uses the encoding of exactly those target-action frames as its training target. Confirm with the teaching team that this is allowed before building it; if it is not, keep Option A only.

#### Scale-Normalized 3-Way Gate (Crucial for Interpretability):

Before fusing history ($h_t$), present ($e_t$), and predicted future ($f_t$), we project and apply LayerNorm to each branch so all three vectors have identical zero mean and unit variance per feature dimension:

$$
\tilde{h}_t^c = \text{LN}(W_h^c h_t^c), \quad \tilde{e}_t^c = \text{LN}(W_e^c e_t^c), \quad \tilde{f}_t^c = \text{LN}(W_f^c f_t^c), \quad c \in \{v, n\}
$$

Compute the 3-way gate weights via a softmax MLP:

$$
[g_1^c, g_2^c, g_3^c] = \text{softmax}\Big(\text{MLP}_g^c\big([\tilde{h}_t^c \,\Vert{}\, \tilde{e}_t^c \,\Vert{}\, \tilde{f}_t^c]\big)\Big)
$$

$$
o_t^c = g_1^c \tilde{h}_t^c + g_2^c \tilde{e}_t^c + g_3^c \tilde{f}_t^c, \quad c \in \{v, n\}
$$

Because $\Vert{}\tilde{h}_t^c\Vert{}_2 \approx \Vert{}\tilde{e}_t^c\Vert{}_2 \approx \Vert{}\tilde{f}_t^c\Vert{}_2$, the scalars $(g_1^c, g_2^c, g_3^c)$ are comparable across branches, and we can plot them per verb and noun class in our final report and poster as an approximate indication of each branch's contribution. They are not a causal importance measure, so pair the plots with branch-removal ablations.

## 6. Loss Functions, Metric Calibration & Cold-Start Priors

### 6.1 Training Loss

Train both AR and STA heads using Class-Balanced Focal Loss (or Cross-Entropy with label smoothing $\epsilon = 0.1$) over the 97 verb and 300 noun classes.

### 6.2 Post-Hoc Metric-Specific Logit Adjustment (Menon et al., 2020)

Let $\pi_v \in \Delta^{96}$ and $\pi_n \in \Delta^{299}$ be the empirical class frequencies in the training set (with Laplace smoothing $\delta = 10^{-4}$). Given raw validation/test logits $z_v \in \mathbb{R}^{97}$ and $z_n \in \mathbb{R}^{300}$:

$$
z_v^{\text{cal}} = \frac{z_v}{T_v} - \tau_v \log \pi_v, \quad z_n^{\text{cal}} = \frac{z_n}{T_n} - \tau_n \log \pi_n
$$

#### Why separate $(\tau_v, \tau_n)$ for AR and STA?

* **For Task 1 (AR):** Primary metric is Action Top-1 Accuracy. Grid-search $(\tau_v^{\text{AR}}, \tau_n^{\text{AR}}) \in [-0.2, 0.3]$ on the validation set to maximize Action Top-1.

* **For Task 2 (STA):** Primary metric is Action MT5R (mean per-class Top-5 recall). Unadjusted models predict common head classes on every clip, giving $0\%$ recall on rare classes. Grid-search $(\tau_v^{\text{STA}}, \tau_n^{\text{STA}}) \in [0.4, 1.2]$ on the validation set to directly maximize Action MT5R.

* **Submission format:** The course defines `verb_output` and `noun_output` as predicted class probabilities, so submit $\text{softmax}(z^{\text{cal}})$ for each, not raw logits. `epic_metrics.save_submission()` checks IDs, shapes and finite values, but not that each row sums to 1.

### 6.3 Solving the $<1.0\text{s}$ Cold-Start Edge Case

For hidden-test events with $t_s < 1.0\text{s}$ (`sta_observation_end_seconds < 0`), no prior video frames exist.

* Compute the empirical distribution of video-initial actions (actions occurring in the first $5.0\text{s}$ of training videos): $\pi_v^{\text{init}}, \pi_n^{\text{init}}$.

* Apply the STA logit adjustment directly to $\log \pi_v^{\text{init}}$ and $\log \pi_n^{\text{init}}$ so that cold-start predictions still maximize Action MT5R across both head and tail opening actions.

## 7. Phased Execution Timeline (Oct 23 $\rightarrow$ Dec 11, 2026)

### Phase 1: Mid-Term Sprint (Oct 23 – Nov 13, 2026)

**Goal:** Lock in a bug-free, leakage-verified baseline, complete VSC feature extraction, and present our first real ablation table at the Mid-Term Presentation on 13/11/2026.

1. **Week 1 (Oct 23 – Oct 30):**

   * Run dataset inspection script on `train.csv`, implement the Hybrid Seen + Unseen Validation Split in `epic_starter/data/manifests.py` (it replaces the current video-grouped split there), and freeze it.

   * Extend the existing leakage checks to the feature-extraction path. `tests/test_epic.py::test_sta_never_samples_gap_or_target` and `validate_batch()` in `scripts/check_data.py` already verify that the data loader never returns STA frames after `sta_observation_end_seconds`.

   * Run ViT-L feature extraction on VSC ($24 \times D$ compact token format) for AR and STA, including the predictor's future tokens (§4.3); see §4 for the backbone and resolution decision.

2. **Week 2 (Oct 31 – Nov 6):**

   * Train AR & STA Baseline (Frozen V-JEPA 2.1 + Factorized Attentive Probe).

   * Implement $<1.0\text{s}$ Cold-Start Prior and $(\tau_v, \tau_n)$ Logit Adjustment grid search.

   * Submit first working `submission.pt` and `submission_sta.pt` ZIP to Codabench.

3. **Week 3 (Nov 7 – Nov 13):**

   * Run Mid-Term ablations (Mean Pool vs. Attentive Probe vs. Factorized Verb/Noun Probe; $\tau=0$ vs. tuned $\tau_v, \tau_n$ on Overall, Unseen, and Tail splits).

   * Prepare Mid-Term Presentation slides (15% of grade).

### Phase 2: Final Research Sprint (Nov 14 – Dec 11, 2026)

**Goal:** Add temporal memory and future prediction one module at a time so every addition becomes a clean row in our final paper's ablation table.

1. **Week 4 (Nov 14 – Nov 20):** Implement single-stream AGA (arXiv:2603.01743) vs. Decoupled Verb-AGA & Noun-AGA.

2. **Week 5 (Nov 21 – Nov 27):** Add LayerNorm-Calibrated 3-Way Gate and test Frozen V-JEPA Predictor ($f_t$) vs. Learned Causal Predictor ($f_t$) ($\text{AR} \rightarrow \text{STA}$ latent regression; only if the rule check in §5.3 allows it).

3. **Week 6–7 (Nov 28 – Dec 11):** Generate gate interpretability plots, finalize clean GitHub repository documentation (25% grade), write conference-format paper (20% grade), design PDF poster (10% grade), and rehearse final defense (30% grade).

## 8. 4-Person Team Responsibilities Matrix

Because individual grades can be adjusted during the defense based on individual contributions, each member owns an end-to-end research component and its corresponding ablation table:

| Team Member | Primary Ownership | Pre-Mid-Term Deliverables (by 13/11) | Post-Mid-Term Deliverables (by 11/12) | Defense & Paper Section | 
| ----- | ----- | ----- | ----- | ----- | 
| **Member 1**   *(Data, Calibration & Evaluation)* | Hybrid Split, Leakage Tests, Logit Adjustment & Cold-Start Priors | Builds Seen/Unseen/Tail split; extends the existing leakage tests to feature extraction; implements $<1.0\text{s}$ prior + $(\tau_v, \tau_n)$ calibration; manages Codabench submissions. | Tunes dynamic Markov action transition priors $P(v_t, n_t \mid \hat{v}_{t-1}, \hat{n}_{t-1})$; runs full Seen/Unseen/Tail breakdown tables. | Section 3 (Splits & Metrics) & Section 5.3 (Long-Tail Calibration Analysis). | 
| **Member 2**   *(VSC Backbone & Factorized Probing — You)* | VSC Feature Extraction & Factorized Spatiotemporal Attentive Probe (AR + STA) | Extracts V-JEPA 2.1 $24\times D$ token bank on VSC; builds AR & STA Factorized Verb/Noun Attentive Probe baseline. | Ablates Global Pool vs. Hand-Crafted Motion Saliency vs. Learned Factorized Verb/Noun Queries on Unseen Participants. | Section 4.1 (Representations & AR/STA Probing) + Spatial Attention Visualizations. | 
| **Member 3**   *(Temporal Memory / AGA)* | Action-Guided Attention (AGA) & Decoupled Verb/Noun Memory | Sets up sequential dataset loader ($S=16$ steps anchored at $t_{\text{obs}}$) and reproduces single-stream AGA baseline. | Implements Decoupled Verb-AGA & Noun-AGA streams; ablates history horizon $S \in \{4, 8, 16, 30\}$. | Section 4.2 (Decoupled Action-Guided Memory) + Horizon Ablations. | 
| **Member 4**   *(Future Prediction & 3-Way Gating)* | Future Latent Estimation ($f_t$) & Normalized 3-Way Gating | Extracts/tests V-JEPA 2.1 predictor tokens $f_t$ (arXiv:2605.20904) with linear/attentive probe to verify predictive signal. | Implements Learned Causal Bridge ($e_{t-S:t} \rightarrow e_{\text{AR}}$, subject to the rule check in §5.3) + LayerNorm 3-Way Gate; plots gate weights $(g_1, g_2, g_3)$ across verbs vs. nouns. | Section 4.3 (Future Latent Bridging & Gate Interpretability Figures). | 

## 9. Verified References

* **AGA (2026):** Tai, T.-M., Casarin, S., Pilzer, A., Nutt, W., & Lanz, O. (2026). Action-Guided Attention for Video Action Anticipation. *ICLR 2026* (arXiv:2603.01743v2).

* **JFAA (2026):** Chu, Q., et al. (2026). JFAA: Technical Report for the EPIC-KITCHENS-100 Action Anticipation Challenge at EgoVis 2026. *CVPR 2026 EgoVis Workshop* (arXiv:2605.20904v1).

* **V-JEPA 2.1 (2026):** Mur-Labadia, L., et al. (2026). V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. (arXiv:2603.14482v3).

* **EPIC-KITCHENS-100 (2020):** Damen, D., et al. (2020). Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100. *IJCV / arXiv:2005.00343*.

* **Logit Adjustment (2020):** Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2020). Long-Tail Learning via Logit Adjustment. *ICLR 2021* (arXiv:2007.07314).

* **AVT (2021):** Girdhar, R., & Grauman, K. (2021). Anticipative Video Transformer. *ICCV 2021* (arXiv:2106.02036).