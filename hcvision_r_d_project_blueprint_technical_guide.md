# KU Leuven HCVISION (2026–2027) — Team Research Blueprint & Technical Guide
**Dataset:** EPIC-KITCHENS-100 (Egocentric Video)  
**Tasks:** Task 1: Action Recognition (AR) $\mid$ Task 2: Short-Term Action Anticipation (STA)  
**Mid-Term Presentation:** 13 November 2026 $\mid$ **Final Submission & Defense:** 11 December 2026

---

## 1. Executive Summary: How We Win the Grade

Before writing code, everyone needs to understand how this project is graded:
* **75% of our grade** comes from scientific communication and defense: Mid-term presentation ($15\%$), Final presentation & defense ($30\%$), Conference-format paper ($20\%$), and Poster ($10\%$).
* **25% of our grade** comes from the solution: leaderboard results, code organization, reproducibility, documentation, and implementation quality.

**What this means for our strategy:**  
The professors are not just looking for whoever burns the most GPU hours on the VSC cluster. They are grading us on whether we **think like researchers**:
1. Did we identify concrete limitations in existing state-of-the-art (SOTA) papers?
2. Did we formulate a clear, mathematically sound hypothesis to fix those limitations?
3. Did we design clean **ablation studies** (stepping-stone experiments) that prove *why* each piece of our model helps across **Seen Participants**, **Unseen Participants**, and **Rare (Tail) Classes**?
4. Because we are a **4-person team** (an exception to the standard 3-person team), the graders will expect a cohesive project where all four members have a clear, defendable research module covering **both Task 1 (AR) and Task 2 (STA)**.

### Our Unified "Mini-Thesis" Title (Working Concept)
> ***Factorized Spatial-Motion Token Disentanglement and Dual-Stream Action-Guided Attention with Predictive Latent Bridging for Egocentric Video Understanding***

In plain English: We take the best 2026 method for temporal memory (**Action-Guided Attention / AGA**) and the best 2026 self-supervised video backbone (**V-JEPA 2.1 / JFAA**), identify **four mathematical and practical bottlenecks** when applying them to EPIC-KITCHENS-100, and introduce a **Spatial-Motion Disentangled Dual-Stream Architecture** that improves both **Action Recognition (AR)** and **Short-Term Action Anticipation (STA)**—especially when tested on **unseen kitchens** and **rare actions**.

---

## 2. Plain-English Guide to the Dataset, Tasks, & Rules

### 2.1 The Dataset: EPIC-KITCHENS-100
* **What it is:** $100$ hours of first-person (head-mounted GoPro) videos of people cooking, washing dishes, and cleaning in their own kitchens across $37$ different participants (`P01` to `P37`).
* **Modality Rule:** Even though the videos have sound, **we are only allowed to use the video frames (no audio)**.
* **The Labels:** Every action is defined as a pair of **(Verb, Noun)**:
  * There are **$97$ Verb classes** (e.g., *open, take, cut, wash, pour, close*).
  * There are **$300$ Noun classes** (e.g., *fridge, milk, knife, sponge, pan, door*).
  * Together, an **Action** is a combined `(verb, noun)` pair (e.g., `(3, 3) = (open, door)`).
* **The "Long-Tail" Problem:** In everyday cooking, a few verbs (*take, put, open*) and nouns (*spoon, plate, tap*) happen thousands of times (**Head classes**), while hundreds of other verbs/nouns (*sharpen, skewer, kiwi, pestle*) happen only a handful of times (**Tail classes**). A naive model will just guess common classes and fail completely on tail classes.

### 2.2 The Two Required Tasks

| Feature | Task 1: Action Recognition (AR) | Task 2: Short-Term Action Anticipation (STA) |
| :--- | :--- | :--- |
| **Goal** | Watch a video clip of an action happening right now and classify what `(verb, noun)` is occurring. | Watch the video **up to $1.0\text{ second}$ before** an action starts, and predict what `(verb, noun)` the person is *about to do next*. |
| **Allowed Video Frames** | Frames inside the action segment $[t_{\text{start}}, t_{\text{end}}]$. | Frames strictly before $t_{\text{obs\_end}} = t_{\text{start}} - 1.0\text{s}$. |
| **Strict Causality Rule** | Standard trimmed clip classification. | **CRITICAL:** You may **never** use any frame inside the $1.0\text{s}$ gap $[t_{\text{start}} - 1.0\text{s}, t_{\text{start}}]$ or during the action itself—not even when extracting backbone features! |
| **Cold-Start Edge Case** | Not applicable (every action has frames). | If an action in the test set starts before $1.0\text{s}$ (e.g., at $0.4\text{s}$), then $t_{\text{obs\_end}} < 0$, meaning **zero video frames are available**. We must output a learned statistical prior without crashing. |
| **Primary Metric** | **Action Top-1 Accuracy** (also reports Verb/Noun Top-1 & Top-5). | **Action Class-Mean Top-5 Recall (MT5R)** (also reports Verb/Noun MT5R). |

### 2.3 Why the Two Metrics Matter (Top-1 Accuracy vs. MT5R)
Understanding the difference between the two metrics is one of our biggest advantages:
1. **AR Metric — Top-1 Accuracy:** Every test clip counts equally. If $20\%$ of the dataset is `take plate`, getting `take plate` right boosts Top-1 Accuracy massively. Thus, **Top-1 Accuracy rewards models that favor common (head) classes**.
2. **STA Metric — Class-Mean Top-5 Recall (MT5R):** First, for each class $c$, we calculate what percentage of true instances of class $c$ had the correct label inside our top $5$ highest-scoring predictions ($\text{Recall@5}_c$). Then, we average $\text{Recall@5}_c$ **equally across all classes present in the test set**:
   $$\text{MT5R} = \frac{1}{|C|} \sum_{c \in C} \text{Recall@5}_c$$
   This means a rare action that occurs only $3$ times in the entire test set has the **exact same weight** in the final STA score as `take plate` (which occurs $2,000$ times)! If our model ignores rare tail classes, our STA MT5R score will tank.

### 2.4 Codabench Submission Format Constraint
For both `submission.pt` (AR) and `submission_sta.pt` (STA), Codabench requires a list of dictionaries—one per test event—containing:
* `narration_id`: The string ID of the clip (e.g., `"P01_101_0"`).
* `verb_output`: A 1D tensor of **$97$ floating-point scores** (logits or probabilities).
* `noun_output`: A 1D tensor of **$300$ floating-point scores** (logits or probabilities).

**Notice:** We **cannot** directly submit $97 \times 300 = 29,100$ joint action scores. The evaluation server computes joint action scores from our marginal $97$-D verb vector and $300$-D noun vector (typically via outer product of probabilities $P(v, n) = P(v) \cdot P(n)$ or sum of log-probabilities). Everything we design must output clean, well-calibrated $97$-D and $300$-D vectors.

---

## 3. Data Engineering: Why We Need a Hybrid Train/Validation Split

The project rules state: *"Use the provided training annotations only for development. Each team chooses and documents its own train/validation split within the provided train set."*

### 3.1 Three Splits We Must Avoid
1. **Trap 1: Random Clip Split (Strictly Forbidden):** If clip `P01_01_0` is in Train and clip `P01_01_1` (2 seconds later in the exact same video) is in Validation, the background frames overlap. Your validation accuracy will look huge ($+15\%$ fake boost) because the model memorized the kitchen counter, and then it will fail on the hidden test set.
2. **Trap 2: Splitting at `P01–P31` (Train) vs. `P32–P37` (Val):**
   * In EPIC-KITCHENS, participants `P01–P32` were recorded in 2018 (EPIC-KITCHENS-55), while `P33–P37` were recorded in 2020 with different camera frame rates and denser annotations.
   * Worse, several rare noun classes only appear in `P32+`. If `P32+` is entirely in validation, our model will see $0$ training examples for those nouns!
3. **Trap 3: Holding Out *Only* Whole Participants for Validation:**
   * The hidden test set evaluates two scenarios: **Seen Participants** (people whose kitchens were in the training set, filmed on a new day) and **Unseen Participants** (brand-new kitchens never seen during training).
   * If our validation set *only* has held-out participants, we have $100\%$ Unseen Participants in validation and $0\%$ Seen Participants. We won't know how our model performs on seen kitchens!

### 3.2 Our Solution: The Hybrid Stratified Validation Split
We will build a deterministic, documented split (`splits/train_split.csv` and `splits/val_split.csv`) comprising **$\approx 85\%$ Train and $\approx 15\%$ Validation**:
1. **Unseen-Participant Val Subset ($\approx 7.5\%$ of data):** Hold out $3$ complete participants spanning both collection waves (e.g., `P02`, `P23`, and `P35`). Zero videos from these $3$ people enter the training set. This gives us an exact local benchmark for the **Unseen Participant** leaderboard metric.
2. **Seen-Participant Val Subset ($\approx 7.5\%$ of data):** For the remaining $34$ participants, sort each person's videos chronologically by video ID (e.g., `P01_01`, `P01_02`, ..., `P01_109`) and assign the **last $10\%$ of whole videos** per person to Validation. Because we split by *whole video files*, zero frames leak between train and val.
3. **Tail-Class Coverage Check:** Run a verification script to ensure every verb and noun present in the validation split has at least $1$ instance in the training split.

---

## 4. Literature Review & Theory (Explained Simply)

Our project combines ideas from four key papers. Here is how each one works in plain English so every teammate can explain it during the defense.

### 4.1 Paper 1: V-JEPA 2.1 & JFAA (Joint-Embedding Predictive Architecture)
* **References:** Bardes et al. (2024), *V-JEPA: Latent Video Prediction for Visual Representation Learning*; JFAA (2026), *EgoVis EPIC-KITCHENS Anticipation Winner*.
* **How Standard Video Models Work:** Most video models either classify frames directly or try to predict the actual RGB pixels of the next frame. Predicting future RGB pixels in first-person video is nearly impossible because the person's head moves constantly, blurring the background.
* **How V-JEPA Works:** Instead of predicting pixels, V-JEPA has two parts pre-trained on massive unlabeled video datasets:
  1. **The Context Encoder $E_{\theta}$:** Takes a short video clip (e.g., $16$ frames over $2\text{ seconds}$) and converts it into a grid of semantic feature vectors (patch tokens) $Z_t \in \mathbb{R}^{N \times D}$, where $N$ is the number of 3D spatiotemporal patches and $D$ is the embedding dimension (e.g., $D = 1024$ for ViT-L).
  2. **The Latent Predictor $P_{\phi}$:** Takes those current feature tokens $Z_t$ alongside positional mask tokens representing a future time window, and predicts what the feature tokens **will look like in the near future**: $\hat{Z}_{t+\Delta} \in \mathbb{R}^{N \times D}$.
* **Why It Is Legal for STA:** Because $P_{\phi}$ only takes frames up to $t_{\text{obs\_end}} = t_{\text{start}} - 1.0\text{s}$ as input and *hallucinates* the future features inside its own neural weights, it strictly obeys the $1.0\text{s}$ gap rule while giving us a "peek" into the $1.0\text{s}$ gap!

### 4.2 Paper 2: AGA (Action-Guided Attention for Video Action Anticipation)
* **Reference:** Tai et al. (2026), *Action-Guided Attention for Video Action Anticipation* (`arXiv:2603.01743`).
* **The Intuition:** Imagine someone is making coffee. Five seconds ago they opened the cupboard; three seconds ago they took a mug; one second ago they picked up the kettle. To predict what they will do next (*pour water into mug*), looking at *what actions they just did* is much cleaner than re-watching $16$ seconds of raw, noisy kitchen video.
* **How AGA Works Step-by-Step:**
  1. **1 FPS Stepping:** We step through the video once per second ($t = 1, 2, \dots, T$). At each second $t$, we have a visual feature vector $e_t \in \mathbb{R}^D$ summarizing that short clip.
  2. **FIFO Memory Queue:** We keep a sliding memory bank of the last $S = 16$ seconds of visual features $V_{t} = [e_{t-S}, \dots, e_{t-1}] \in \mathbb{R}^{S \times D}$ and the model's own past action probability predictions $P_{t} = [\hat{p}_{t-S}, \dots, \hat{p}_{t-1}] \in \mathbb{R}^{S \times C}$.
  3. **Predictions as Queries and Keys:** In standard Transformer attention, Queries ($Q$), Keys ($K$), and Values ($V$) all come from visual features. In AGA:
     * **Query ($q_t$):** An Exponential Moving Average (EMA) of the model's recent action predictions:
       $$\tilde{p}_t = \alpha \hat{p}_t + (1 - \alpha) \tilde{p}_{t-1}, \quad q_t = W_Q \tilde{p}_t$$
     * **Keys ($K_t$):** The past $S$ action predictions projected by $W_K$:
       $$K_t = P_t W_K$$
     * **Values ($V_t$):** The past $S$ visual features projected by $W_V$:
       $$V_t = E_t W_V$$
  4. **Retrieving History ($h_t$):** The model compares its current action state ($q_t$) against past action states ($K_t$) to pull out the most relevant past visual features ($V_t$):
     $$h_t = \text{Softmax}\left(\frac{q_t K_t^\top}{\sqrt{d_k}}\right) V_t$$
  5. **Gated Fusion:** An adaptive gate mixes the retrieved history $h_t$ with the current visual evidence $e_t$ to predict the action $1\text{ second}$ in the future.

### 4.3 Paper 3: Egocentric Hand-Object Disentanglement (ORViT & InAViT)
* **References:** Herzig et al. (2022), *Object-Region Video Transformers (ORViT)*; Roy et al. (2024), *Interaction Region Visual Transformer for Egocentric Action Anticipation (InAViT)*.
* **The Problem with Global Video Features:** In a $640 \times 360$ kitchen video, $80\%$ of the pixels are static cabinets, walls, appliances, and floors. Only $20\%$ of the image contains the person's hands and the object they are touching. If you average-pool all patch tokens together, $80\%$ of your vector represents the kitchen background!
* **Why This Hurts Unseen Participants:** When tested on a Seen Participant, the model recognizes the kitchen wallpaper and guesses what that person usually cooks. When tested on an **Unseen Participant**, the kitchen background changes completely, causing the model to fail.
* **The Core Insight:**
  * **Verbs** (*cut, stir, wash, open, shake*) are defined by **temporal motion trajectories** (how the hands and active region move between frames).
  * **Nouns** (*carrot, spoon, sponge, drawer, bottle*) are defined by **local foreground object appearance** (the sharp visual patches right where the motion is happening).

### 4.4 Paper 4: Long-Tail Logit Adjustment
* **Reference:** Menon et al. (2020), *Long-Tail Learning via Logit Adjustment*.
* **The Math:** Suppose a neural network outputs raw score logits $s_y(x)$ for class $y$, and the true frequency (prior probability) of class $y$ in the training set is $\pi_y = \frac{N_y}{N_{\text{total}}}$.
  * To maximize **Top-1 Accuracy (Task 1: AR)**, you want standard Bayes-optimal predictions that respect the natural frequency of common classes.
  * To maximize **Balanced / Class-Mean Recall (Task 2: STA MT5R)**, where rare classes count just as much as common classes, Menon et al. prove mathematically that the optimal decision rule shifts the logits by subtracting the scaled log-prior:
    $$s_y^{\text{adjusted}}(x) = s_y(x) - \tau \log(\pi_y + \epsilon)$$
    When $\tau > 0$, rare classes (which have a very negative $\log \pi_y$) get a positive boost, pulling them into the Top-5 predictions and directly increasing **MT5R**!

---

## 5. Four Flaws in the Initial Draft Proposal (And How We Fix Them)

Our initial proposal suggested combining frozen V-JEPA 2.1 + AGA with a 3-way gate. That is a fantastic starting point, but if we implement it naively as written, we will hit **four major roadblocks**:

### Flaw 1: The "Mean Pool vs. Attentive Probe" Contradiction
* **What the draft said:** *"Extract features once, store pooled vectors only"* AND *"Use JFAA's attentive probe with separate verb/noun queries."*
* **Why it breaks mathematically:** JFAA's attentive probe uses cross-attention where learnable Verb and Noun query vectors attend over a set of $N$ spatial-temporal tokens to select the hands and objects. If you mean-pool all tokens into a **single vector** $e_t \in \mathbb{R}^{1 \times D}$ before saving to disk, **attention over a sequence of length $1$ always assigns a weight of $1.0$ to that single vector!** The attentive probe does nothing at all, and $80\%$ of your vector is polluted by static kitchen background.
* **Our Fix (Detector-Free Spatial-Motion Token Extraction):** During VSC feature extraction, before discarding the patch grid $Z_t \in \mathbb{R}^{T' \times H' \times W' \times D}$, we use a $5$-line tensor operation to extract three tiny, specialized representations per clip:
  1. $e_t^{\text{glob}} \in \mathbb{R}^{1 \times D}$: Global average pool (scene context).
  2. $e_t^{\text{mot}} \in \mathbb{R}^{1 \times D}$: Motion-salient pool (weighted by temporal token change between frames—captures hand/action motion for **Verbs**).
  3. $E_t^{\text{obj}} \in \mathbb{R}^{K \times D}$ (with $K = 8$): The top-$8$ spatial patch tokens located around the high-motion interaction region (captures sharp foreground objects for **Nouns** and enables a true **Attentive Probe**!).

### Flaw 2: Concatenating 397-D Verb+Noun Predictions into One AGA Stream
* **What the draft said:** Concatenate the $97$ verb probabilities and $300$ noun probabilities into a single $397$-D vector to drive a single AGA attention layer and a single 3-way gate.
* **Why it is suboptimal:** In cooking, **verbs and nouns follow completely different temporal patterns**:
  * **Noun Persistence:** A person does `take potato` $\rightarrow$ `wash potato` $\rightarrow$ `peel potato` $\rightarrow$ `cut potato`. The **noun** (`potato`) stays identical for $30\text{ seconds}$, while the **verb** changes every $3\text{ seconds}$!
  * **Verb Transitions:** Verbs follow rapid physical sequences (`open` $\rightarrow$ `take` $\rightarrow$ `close`).
* **Our Fix (Factorized Dual-Stream AGA):** We split AGA into two parallel, lightly coupled branches: a **Verb-AGA Stream** (driven by $97$-D verb predictions attending over motion features $e_t^{\text{mot}}$) and a **Noun-AGA Stream** (driven by $300$-D noun predictions attending over object-probed features $e_t^{\text{obj}}$), each with its own specialized 3-Way Gate!

### Flaw 3: Task 1 (Action Recognition) Had Zero Research Novelty
* **What the draft said:** Spend $95\%$ of the effort on STA, and for AR just run a simple probe on mean-pooled features.
* **Why it hurts our grade:** Both AR and STA are mandatory tasks graded equally on the benchmark.
* **Our Fix:** Because we extract our disentangled tokens ($e_t^{\text{glob}}, e_t^{\text{mot}}, E_t^{\text{obj}}$) across the dataset, we build a **Disentangled Cross-Attention Action Recognizer (DC-AR)** for Task 1 where Verb Queries attend to motion+global tokens and Noun Queries attend to the $K=8$ foreground object tokens, plus we use our trained AR embeddings as a **teacher target** to supervise the V-JEPA future predictor in STA!

### Flaw 4: Ignoring Metric Calibration & Cold-Start Priors
* **Our Fix:** We add a **Compositional Logit Calibration & Markov Transition Prior** module that optimizes Top-1 Accuracy for AR ($\tau_{\text{AR}} \approx 0$) and MT5R for STA ($\tau_{\text{STA}} > 0$), while using the empirical verb-noun distribution to cleanly solve test events starting before $1.0\text{ second}$.

---

## 6. Step-by-Step Mathematical & Architectural Specification

Here is the exact pipeline from raw video on VSC to our final Codabench submission files.

```
========================================================================================
STAGE 1: ONE-TIME VSC FEATURE EXTRACTION (Frozen V-JEPA 2.1 ViT-L, 1s Stride)
========================================================================================
Raw 2s Video Window [t - 2s, t] (16 frames @ 256x256)
   │
   ├──► Frozen V-JEPA Encoder E_θ ──► Patch Grid Z_t ∈ R^(T' × H' × W' × D)
   │       ├── 1. Mean Pool over all patches ──────────────► e_t^glob ∈ R^D (Scene Context)
   │       ├── 2. Temporal Diff Saliency Pool ||ΔZ_t||_2 ──► e_t^mot  ∈ R^D (Hand/Verb Motion)
   │       └── 3. Top-K Active Patch Selection (K=8) ──────► E_t^obj  ∈ R^(8 × D) (Object Patches)
   │
   └──► Frozen V-JEPA Predictor P_φ (predicting [t, t + 1s] latent tokens)
           ├── 1. Mean Pool over predicted tokens ─────────► f_t^glob ∈ R^D
           └── 2. Saliency-Weighted Pool ──────────────────► f_t^mot  ∈ R^D, f_t^obj ∈ R^D

========================================================================================
STAGE 2: TASK 1 — ACTION RECOGNITION (AR) ON TRIMMED CLIPS [t_start, t_end]
========================================================================================
Clip Tokens {e^glob, e^mot, E^obj} inside action segment
   ├──► Verb Cross-Attention Probe (Query q_v attends to Temporal Motion e^mot & e^glob) ──► Verb Logits (97-D)
   └──► Noun Cross-Attention Probe (Query q_n attends to K=8 Object Tokens E^obj)        ──► Noun Logits (300-D)

========================================================================================
STAGE 3: TASK 2 — SHORT-TERM ANTICIPATION (STA) UP TO t_obs_end = t_start - 1.0s
========================================================================================
At each 1s step t ≤ t_obs_end:
   ├──► [VERB STREAM]
   │      • Past 97-D Verb Probs (Q_v, K_v) + Past Motion Features (V_v = e_{t-S:t-1}^mot) ──► History h_t^v
   │      • Verb 3-Way Gate: o_t^v = g_1^v * h_t^v + g_2^v * e_t^mot + g_3^v * f_t^mot
   │      • Verb Classifier Head ──► Raw Verb Logits s_t^v (97-D)
   │
   └──► [NOUN STREAM]
          • Attentive Object Probe over E_t^obj (K=8 patches) ──► Present Object Vector e_t^obj
          • Past 300-D Noun Probs (Q_n, K_n) + Past Object Vectors (V_n = e_{t-S:t-1}^obj) ──► History h_t^n
          • Noun 3-Way Gate: o_t^n = g_1^n * h_t^n + g_2^n * e_t^obj + g_3^n * f_t^obj
          • Noun Classifier Head ──► Raw Noun Logits s_t^n (300-D)

========================================================================================
STAGE 4: METRIC-AWARE LOGIT CALIBRATION & COLD-START HANDLER
========================================================================================
   • For AR (Top-1 Metric): Mild temperature scaling + Markov Action Prior compatibility check.
   • For STA (MT5R Metric): Long-Tail Logit Adjustment s_adj = s - τ * log(π) to boost rare tail classes.
   • For STA Events with t_start < 1.0s: Bypass visual pipeline; output calibrated initial-action prior π_init.
```

---

### 6.1 Stage 1 Details: Detector-Free Spatial-Motion Token Disentanglement
When we pass a $16$-frame clip ending at second $t$ through frozen V-JEPA ViT-L ($D = 1024$), the encoder outputs a 4D grid of patch tokens:
$$Z_t \in \mathbb{R}^{T' \times H' \times W' \times D}$$
where $T' = 8$ (temporal tubelets of $2$ frames each) and $H' \times W' = 16 \times 16 = 256$ spatial patches per time step.

Instead of blindly averaging all $8 \times 256 = 2048$ tokens into one vector, we compute:
1. **Global Scene Context Vector ($e_t^{\text{glob}} \in \mathbb{R}^D$):**
   $$e_t^{\text{glob}} = \frac{1}{T' H' W'} \sum_{\tau=1}^{T'} \sum_{h=1}^{H'} \sum_{w=1}^{W'} Z_{t, \tau, h, w}$$
2. **Motion-Salient Vector for Verbs ($e_t^{\text{mot}} \in \mathbb{R}^D$):**
   How do we find where the hands and active tools are moving without running an expensive object detector? We measure how much each spatial patch $(h, w)$ changes between consecutive time steps $\tau - 1$ and $\tau$:
   $$m_{\tau, h, w} = \left\| \text{LayerNorm}(Z_{t, \tau, h, w}) - \text{LayerNorm}(Z_{t, \tau-1, h, w}) \right\|_2 \quad \text{for } \tau \in \{2, \dots, T'\}$$
   Static background patches (cabinets, walls) change uniformally or minimally after feature normalization, whereas patches containing moving hands or manipulated objects exhibit high semantic feature change $m_{\tau, h, w}$. We convert $m_{\tau, h, w}$ into a spatial-temporal probability mask via temperature-scaled Softmax ($\beta = 0.1$):
   $$w_{\tau, h, w}^{\text{mot}} = \frac{\exp(m_{\tau, h, w} / \beta)}{\sum_{\tau', h', w'} \exp(m_{\tau', h', w'} / \beta)}$$
   $$e_t^{\text{mot}} = \sum_{\tau=2}^{T'} \sum_{h=1}^{H'} \sum_{w=1}^{W'} w_{\tau, h, w}^{\text{mot}} Z_{t, \tau, h, w}$$
3. **Top-$K$ Active Object Tokens for Nouns ($E_t^{\text{obj}} \in \mathbb{R}^{K \times D}$, with $K = 8$):**
   At the final observed tubelet $\tau = T'$ (closest to the action), we rank all $256$ spatial patches $(h, w)$ by their motion-saliency score $m_{T', h, w}$ combined with their $L_2$ feature norm, and select the **top $K = 8$ spatial patch tokens**:
   $$E_t^{\text{obj}} = \text{TopK}_{h, w}\left(Z_{t, T', h, w}\right) \in \mathbb{R}^{8 \times D}$$
   **Storage Cost Check:** Saving $e_t^{\text{glob}}$ ($1\times D$), $e_t^{\text{mot}}$ ($1\times D$), $f_t^{\text{mot}}$ ($1\times D$), $f_t^{\text{obj}}$ ($1\times D$), and $E_t^{\text{obj}}$ ($8\times D$) in `float16` takes only $12 \times 1024 \times 2\text{ bytes} \approx 24\text{ KB}$ per second of video! For all $100\text{ hours}$ ($360,000\text{ seconds}$) of EPIC-KITCHENS-100, the entire pre-extracted dataset is **only $\approx 8.6\text{ GB}$ on disk**—meaning it loads into RAM in seconds!

---

### 6.2 Stage 2 Details: Task 1 (Action Recognition) — Disentangled Cross-Attention Probe
For an Action Recognition segment spanning $[t_{\text{start}}, t_{\text{end}}]$, we sample $M$ clips inside the segment (e.g., $M = 4$ evenly spaced clips).
* **Verb Recognition Branch:** Learnable Verb Queries $Q_{\text{AR}}^v \in \mathbb{R}^{N_q \times D}$ apply multi-head cross-attention over the sequence of motion and global vectors $\{e_m^{\text{mot}}, e_m^{\text{glob}}\}_{m=1}^M$ to produce the clip's verb embedding $z_{\text{AR}}^v \in \mathbb{R}^D$ and $97$-D verb logits.
* **Noun Recognition Branch:** Learnable Noun Queries $Q_{\text{AR}}^n \in \mathbb{R}^{N_q \times D}$ apply multi-head cross-attention over all $M \times 8 = 32$ foreground object tokens $\{E_m^{\text{obj}}\}_{m=1}^M$ to isolate the manipulated kitchen object, producing the noun embedding $z_{\text{AR}}^n \in \mathbb{R}^D$ and $300$-D noun logits.
* **Bonus Synergy with STA (AR-to-STA Latent Supervision):** Once our AR model is trained, for every future action in the training set, we can pass the true future action's $z_{\text{AR}}^v$ and $z_{\text{AR}}^n$ as an auxiliary cosine-similarity target for our STA model's future features $f_t^{\text{mot}}$ and $f_t^{\text{obj}}$!

---

### 6.3 Stage 3 Details: Task 2 (STA) — Factorized Dual-Stream AGA + 3-Way Gate
At each $1\text{ second}$ step $t \le t_{\text{obs\_end}} = t_{\text{start}} - 1.0\text{s}$:

1. **Object Token Probing:** First, our learnable Noun Query $q_{\text{probe}}^n \in \mathbb{R}^{1 \times D}$ attends over the $K = 8$ foreground object patches $E_t^{\text{obj}} \in \mathbb{R}^{8 \times D}$ to extract a sharp single object vector for step $t$:
   $$e_t^{\text{obj}} = \text{CrossAttn}\left(q_{\text{probe}}^n, E_t^{\text{obj}}, E_t^{\text{obj}}\right) \in \mathbb{R}^D$$
2. **Dual-Stream Action-Guided Attention (Verb-AGA & Noun-AGA):**
   Instead of lumping verbs and nouns together, we maintain two separate FIFO memory queues of length $S = 16$ seconds:
   * **Verb Memory:** Past verb probability distributions $P_t^v \in \mathbb{R}^{S \times 97}$ act as Keys ($K_t^v = P_t^v W_K^v$), the EMA of verb predictions $\tilde{p}_t^v \in \mathbb{R}^{97}$ acts as the Query ($q_t^v = \tilde{p}_t^v W_Q^v$), and past motion features $E_{t-S:t-1}^{\text{mot}} \in \mathbb{R}^{S \times D}$ act as Values ($V_t^v$):
     $$h_t^v = \text{Softmax}\left(\frac{q_t^v (K_t^v)^\top}{\sqrt{d_k}}\right) V_t^v$$
   * **Noun Memory:** Past noun probability distributions $P_t^n \in \mathbb{R}^{S \times 300}$ act as Keys ($K_t^n = P_t^n W_K^n$), the EMA of noun predictions $\tilde{p}_t^n \in \mathbb{R}^{300}$ acts as the Query ($q_t^n = \tilde{p}_t^n W_Q^n$), and past object features $E_{t-S:t-1}^{\text{obj}} \in \mathbb{R}^{S \times D}$ act as Values ($V_t^n$):
     $$h_t^n = \text{Softmax}\left(\frac{q_t^n (K_t^n)^\top}{\sqrt{d_k}}\right) V_t^n$$
   *(Optional Cross-Cue Coupling: We also test a lightweight linear projection allowing the previous verb state to inform the noun query and vice versa, since knowing the verb was "open" helps predict the noun "fridge" or "cupboard".)*
3. **Separate 3-Way Temporal Gates for Verbs and Nouns:**
   For the Verb stream, an MLP looks at the concatenated History ($h_t^v$), Present Motion ($e_t^{\text{mot}}$), and V-JEPA Imagined Future ($f_t^{\text{mot}}$) and outputs three positive weights that sum to $1$:
   $$[g_1^v, g_2^v, g_3^v] = \text{Softmax}\left(\text{MLP}_v\left([h_t^v \;\|\; e_t^{\text{mot}} \;\|\; f_t^{\text{mot}}]\right)\right)$$
   $$o_t^v = g_1^v h_t^v + g_2^v e_t^{\text{mot}} + g_3^v f_t^{\text{mot}}$$
   Similarly, the Noun stream computes its own gate weights $[g_1^n, g_2^n, g_3^n]$ to fuse $h_t^n$, $e_t^{\text{obj}}$, and $f_t^{\text{obj}}$ into $o_t^n$.
4. **Why This Is Awesome for Our Paper & Defense:**
   We can plot the average values of $(g_1^v, g_2^v, g_3^v)$ versus $(g_1^n, g_2^n, g_3^n)$ across the validation set! We hypothesize that **Verbs** will assign a higher weight to the V-JEPA future predictor ($g_3^v$) and present motion ($g_2^v$), whereas **Nouns** will assign a much higher weight to past history ($g_1^n$) and present object patches ($g_2^n$) because kitchen objects remain in the user's hands across multiple consecutive actions.

---

### 6.4