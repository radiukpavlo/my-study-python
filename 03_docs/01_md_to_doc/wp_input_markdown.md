# WEDD-RST: An Interpretable Framework for Defect Detection in Solar Cyber-Physical Systems

Pavlo Radiuk$^{1,*}$, Anatoliy Sachenko$^{2,3}$, Oleksandr Melnychenko$^{1}$, Antonina Kashtalian$^{1}$

$^1$ Khmelnytskyi National University, 11, Instytutska Str., Khmelnytskyi, 29016, Ukraine
$^2$ Casimir Pulaski Radom University, 29, Malczewskiego str., Radom, 26-600, Poland
$^3$ West Ukrainian National University, 11, Lvivska str., Ternopil, 46009, Ukraine

$^*$ Corresponding author.

---

### Abstract

The rapid proliferation of cyber-physical systems in renewable energy infrastructure has generated vast quantities of continuous sensor data, necessitating advanced monitoring solutions to mitigate safety risks such as photovoltaic fires. However, prevailing deep learning approaches for defect detection often operate as opaque black boxes, lacking the reasoning transparency required for safety-critical decision-making and regulatory compliance. In this work, a novel production knowledge base methodology is proposed for constructing interpretable, rule-based classification systems directly from continuous sensor measurements. The approach integrates rough set theory with a newly developed Weighted Entropy-Density Discretization method to generate production rules that are simultaneously accurate, concise, and human-readable. By optimizing discretization thresholds based on a dual criterion of information entropy and local probability density, the system recovers natural data structures that standard methods often miss. Validated on a simulated SCADA dataset for fire hazard detection, the methodology achieves 96.2% overall accuracy, with a perfect 100% accuracy for deterministic rules and a macro F1-score of 0.960. Notably, the critical Fire Hazard class achieved a 98.0% recall rate, ensuring high reliability in emergency scenarios. Further benchmarking on the Iris and Wine datasets demonstrates competitive performance (93.3% and 87.6% accuracy, respectively) against Decision Tree and Naive Bayes baselines. The resulting knowledge base is self-contained and lightweight, enabling deployment on resource-constrained edge devices where it provides fully auditable IF-THEN rules with explicit confidence scores, effectively bridging the gap between high-performance artificial intelligence and the rigorous requirements for transparent industrial automation.

Keywords: production knowledge base, rough set theory, discretization, information granulation, reduct, classification, cyber-physical systems, photovoltaic monitoring, SCADA

---

## 1. Introduction

The proliferation of cyber-physical systems (CPS) in industrial environments, particularly in renewable energy infrastructure, has generated unprecedented volumes of continuous sensor data that require intelligent analysis for real-time decision-making. Photovoltaic (PV) power stations represent a critical application domain where thermal defects in modules, including overheating bypass diodes and localized hot spots, can escalate to catastrophic fire incidents if not detected and classified promptly [1], [2]. Modern PV monitoring architectures employ Supervisory Control and Data Acquisition (SCADA) systems that integrate diverse sensor modalities, including thermal imaging from Unmanned Aerial Vehicle (UAV) platforms, bypass diode temperature sensors, and fire detection sensors [3], [4].

The fundamental challenge in these cyber-physical environments is the transformation of raw continuous sensor measurements into actionable knowledge that operators can interpret, verify, and trust. While deep learning methods such as You Only Look Once (YOLO)-family architectures have demonstrated impressive detection accuracy [5], [6], they operate as nontransparent "black boxes" that cannot explain their reasoning. In safety-critical domains such as fire hazard assessment, this opacity creates regulatory, operational, and ethical concerns [7]. An operator receiving a fire hazard alert needs to understand why the system classified a condition as dangerous—not merely accept a probability score from an inscrutable neural network.

The task of building production rules from sensor data is a classic problem of inductive learning, i.e., machine learning based on symbolic computing [8], [9]. Given a feature matrix $B$ (where rows represent measurement samples and columns represent sensor features) and a class label vector $Y$ (operational states), the goal is to construct logical rules of the form:

> IF (Feature$_1 \in$ Range$_1$) AND (Feature$_2 \in$ Range$_2$) $\ldots$ THEN Class $= k$

with associated metrics of confidence (certainty) and support (statistical significance). This transition from tabular data to logical constructions requires addressing several interconnected challenges: discretization of continuous features into meaningful categories, identification of redundant features, resolution of contradictory patterns, and design of inference mechanisms for new observations.

Classical approaches to this problem include decision tree algorithms, such as ID3 [9], C4.5 [10], and CART [11], sequential covering algorithms (CN2, PRISM, RIPPER), and Rough Set Theory (RST) methods [12]. Each has notable limitations:

Decision tree algorithms construct rules greedily, selecting locally optimal splits at each node without guaranteeing globally optimal rule sets. The resulting rules may be unnecessarily long or miss important feature interactions [13]. Sequential covering algorithms build rules directly but are sensitive to the order of class selection and may produce overlapping or contradictory rules.

Standard RST methods provide rigorous mathematical foundations for handling uncertainty through lower and upper approximations [8], [14] but traditionally rely on simple equal-width or equal-frequency discretization that ignores the natural structure of the data. The Fayyad-Irani entropy-based discretization [15], while effective for class separation, does not consider the density distribution of feature values, leading to cut points that bisect natural data clusters and produce rules sensitive to noise.

This paper proposes a Production Knowledge Base (PKB) Methodology that addresses these limitations through a hybrid approach combining enhanced discretization with RST. The key scientific contributions are:

1Weighted Entropy-Density Discretization (WEDD): A novel multi-criteria discretization method that simultaneously minimizes information entropy with respect to class labels and local probability density at threshold positions. This dual-criterion approach places cut points in natural "valleys" between data clusters, producing rules that are robust to statistical outliers.
2An improved heuristic for finding minimal feature subsets (reducts) that incorporates information-capacity weights derived from the discretization stage, reducing rule complexity by up to 40–60% while preserving classification capability.
3A conflict resolution mechanism based on an integral class support score that combines rule confidence, support, and interval reliability weights, enabling accurate classification even when multiple contradictory rules are activated.

The remainder of this paper is organized as follows: Section 2 surveys related work. Section 3 presents the complete mathematical framework of the PKB methodology. Section 4 reports experimental results. Section 5 discusses findings and implications. Section 6 concludes the paper.

## 2. Related Works

### 2.1 Rough Set Theory for Knowledge Discovery

RST, introduced by Pawlak in 1982 [12], provides a mathematical framework for reasoning about imprecise, uncertain, and incomplete data without requiring external parameters such as probability distributions or fuzzy membership functions. The theory is built upon the concept of indiscernibility relations, which partition a universe of objects into equivalence classes based on their attribute values [8].

The key mathematical constructs, such as lower approximation, upper approximation, and boundary region, enable the extraction of deterministic and probabilistic knowledge from data [14]. Skowron and Rauszer [16] formalized the discernibility matrix and discernibility function, providing the theoretical basis for finding minimal feature subsets (reducts) that preserve classification capability. The edited volume by Słowiński [17] established the practical foundations of RST for intelligent decision support, while Greco et al. [18] extended the framework to multicriteria decision analysis.

Pawlak's later work [19] explicitly connected RST to data analysis, establishing decision tables, decision rules, and reducts as the primary constructs for knowledge extraction. The LERS system [20] demonstrated practical rule learning from rough sets, laying groundwork for PKB construction. Despite these advances, most RST implementations rely on pre-discretized data, with limited attention to the quality and methodology of the discretization step itself.

### 2.2 Feature Discretization Methods

Discretization, i.e., the transformation of continuous features into discrete intervals, is a critical preprocessing step for symbolic classification methods [21]. The seminal work of Toulabinejad et al. [15] utilized entropy-based recursive discretization using the Minimum Description Length Principle (MDLP) as a stopping criterion, which remains the *de facto* standard for supervised discretization. Their method selects split points that minimize the conditional entropy of class labels, effectively finding thresholds that best separate classes.

García et al. [21] provided a comprehensive taxonomy of discretization methods, categorizing them along dimensions of supervision (supervised vs. unsupervised), splitting strategy (top-down vs. bottom-up), and evaluation criterion (entropy, error, statistical). Dougherty et al. [22] demonstrated that supervised discretization significantly improves the performance of symbolic classifiers compared to unsupervised binning. Liu et al. [23] studied discretization as an enabling technique, establishing the theoretical connections between discretization quality and downstream classification accuracy.

A significant limitation of entropy-only methods is their disregard for the density distribution of feature values. As Xun et al. [24] showed in their comparative study of entropy-based approaches, different discretization criteria can lead to substantially different cut points and classification outcomes. This observation motivates the multi-criteria approach presented here, which augments entropy with density information.

### 2.3 Kernel Density Estimation

Kernel Density Estimation (KDE), independently proposed by Rosenblatt [25] and Parzen [26], provides a nonparametric method for estimating probability density functions from data. Silverman's monograph [27] established the practical foundations of KDE, including bandwidth selection methods such as Scott's rule. In the context of discretization, KDE enables identification of "natural breaks" in data distributions, i.e., valleys between density peaks that represent meaningful boundaries between subpopulations. The WEDD method leverages this capability to place discretization thresholds at density minima, ensuring that the resulting intervals respect the inherent structure of the data.

### 2.4 Cyber-Physical Systems for PV Monitoring

Modern PV monitoring systems employ edge-cloud architectures that combine UAV-based thermal imaging with SCADA integration for fire hazard detection [2]. Deep learning approaches, particularly YOLO-family architectures [5], have been applied to thermal defect detection in PV modules [1]. Thakfan et al. [3] provided a comprehensive review of artificial intelligence (AI)-based fault detection and diagnosis methods for building energy systems, highlighting the trade-off between detection accuracy and interpretability.

Boolean logic-based decision systems in SCADA environments correlate multiple sensor signals, such as thermal anomalies, bypass diode temperatures, and fire detection sensors, to classify operational states [28]. The integration of interpretable production rules with such SCADA logic represents an opportunity to enhance the transparency and auditability of automated fire hazard assessment, which is the motivating application for the methodology presented in this paper.

## 3. Methodology

The PKB Methodology consists of a seven-stage pipeline that transforms raw continuous sensor data into a self-contained, interpretable classification system. Figure 1 illustrates the complete pipeline architecture.

![Complete methodology pipeline of the PKB system. The seven stages are color-coded by phase: blue stages (1–2) perform data preprocessing and discretization; green stages (3–5) apply RST for knowledge extraction; orange stages (6–7) assemble the knowledge base and execute inference. Data flows from left to right, with each stage's output serving as input to the next.](figs/fig_methodology_pipeline.png)

### 3.1 Problem Formalization: The Decision-Making System

Definition 1 (Decision-Making System). The information model of the classification task is represented as a Decision-Making System (DMS), described by the tuple:

$$ \mathrm{DMS} = \langle U, A \cup \{d\}, V, f \rangle \tag{1} $$

where $U = \{x_1, x_2, \ldots, x_n\}$ is a finite set of objects (universe, where each object $x_i$ corresponds to a row of the feature matrix $B$); $A = \{a_1, a_2, \ldots, a_m\}$ is a set of conditional attributes (features) corresponding to the columns of $B$; $d$ is the decision attribute, determined by the class label vector $Y$; $V = \bigcup_{a \in A \cup \{d\}} V_a$ is the domain of attribute values; $f : U \times (A \cup \{d\}) \rightarrow V$ is an information function mapping objects to their attribute values.

Figure 2 provides a visual representation of the DMS formalization, showing how the feature matrix $B$ and class vector $Y$ map to the formal tuple representation.

![Formal structure of the DMS. The feature matrix B provides conditional attributes A, the class vector Y provides the decision attribute d, and the information function f maps objects to their attribute values in domain V.](figs/fig_dms_model.png)

### 3.2 Stage 1: Multi-Criteria Adaptive Discretization (WEDD)

Since the feature matrix $B$ contains continuous values, they must be partitioned into intervals to enable the construction of symbolic production rules. The WEDD method is proposed, which simultaneously optimizes two criteria: information entropy (class separation) and local probability density (natural data structure).

#### Criterion 1: Information Entropy

For attribute $a \in A$ and a candidate threshold $T$, the classical conditional entropy is computed as:

$$ E(a, T; U) = \frac{|U_{\text{left}}|}{|U|} \mathrm{Ent}(U_{\text{left}}) + \frac{|U_{\text{right}}|}{|U|} \mathrm{Ent}(U_{\text{right}}) \tag{2} $$

where $U_{\text{left}} = \{x \in U : f(x, a) < T\}$ and $U_{\text{right}} = \{x \in U : f(x, a) \geq T\}$ are the subsets of objects split by threshold $T$, and $\mathrm{Ent}(S)$ denotes the Shannon entropy [29] of the class distribution within subset $S$:

$$ \mathrm{Ent}(S) = -\sum_{k=1}^{K} p_k \log_2 p_k \tag{3} $$

with $p_k$ being the proportion of class $k$ in $S$ and $K$ the total number of classes.

#### Criterion 2: Local Density Distribution

To estimate the density distribution of attribute values, KDE [25], [26] is employed:

$$ \hat{f}_a(v) = \frac{1}{n \cdot h} \sum_{i=1}^{n} K\left(\frac{v - b_{ia}}{h}\right) \tag{4} $$

where $b_{ia}$ is the value of attribute $a$ for object $x_i$, $K(\cdot)$ is a Gaussian kernel function, and $h$ is the bandwidth parameter determined by Scott's rule [27]. The key insight is that optimal discretization thresholds should pass through regions of low density, the so-called "valleys" between natural clusters, rather than arbitrarily bisecting dense regions.

#### Multi-Criteria Objective Functional

An integral quality criterion $Q(a, T)$ for threshold optimization is introduced and must be minimized:

$$ Q(a, T) = \alpha \cdot \bar{E}(a, T) + (1 - \alpha) \cdot \bar{f}_a(T) \tag{5} $$

where $\bar{E}(a, T)$ is the min-max normalized entropy, $\bar{f}_a(T)$ is the min-max normalized density at point $T$, and $\alpha \in [0, 1]$ is a weighting coefficient controlling the trade-off between class separation and structural preservation. In the conducted experiments, $\alpha = 0.6$ emphasizes class separation while still incorporating density information.

#### Recursive Discretization Algorithm

The WEDD algorithm proceeds as follows:

```text
Algorithm: WEDD Discretization
Input: Feature column b_a, class vector Y, weight alpha
Output: Set of optimal thresholds T*_a

1: Sort values b_{ia} in ascending order
2: Generate candidate thresholds T as midpoints between adjacent distinct values
3: Compute KDE profile \hat{f}_a(v) using Gaussian kernel with Scott bandwidth

4: For each candidate T in T:
5:     Compute conditional entropy E(a, T; U) via Equation (2)
6:     Evaluate density \hat{f}_a(T) via Equation (4)
7: End For

8: Normalize:
       \bar{E}(a, T) = (E - E_min) / (E_max - E_min)
       \bar{f}_a(T) = (\hat{f}_a - f_min) / (f_max - f_min)

9: Compute Q(a, T) = alpha * \bar{E} + (1 - alpha) * \bar{f}_a for all T
10: Select T* = argmin_T Q(a, T)
11: Apply MDLP stopping criterion

12: If MDLP criterion is not met:
13:     Recurse on sub-intervals U_left and U_right
14: End If

15: Return T*_a = {tau_1, tau_2, ..., tau_ka}
```

### 3.3 Stage 2: Feature Space Transformation and Information Granulation

After discretization, continuous values in $B$ are mapped to discrete codes using a quantization function.

Definition 2 (Quantization Function). For each attribute $a_j \in A$ with optimal thresholds $T^*_j = \{\tau_{j,1}, \ldots, \tau_{j,k_j}\}$, the quantization function $\phi_j : V_{a_j} \rightarrow \mathbb{Z}^+$ maps continuous values to interval codes:

$$ b^*_{ij} = \phi_j(b_{ij}) =
\begin{cases}
    0, & \text{if } b_{ij} < \tau_{j,1}; \\
    m, & \text{if } \tau_{j,m} \leq b_{ij} < \tau_{j,m+1}; \\
    k_j, & \text{if } b_{ij} \geq \tau_{j,k_j}.
\end{cases} \tag{6} $$

The result is a discrete matrix $B^* = \{b^*_{ij}\}$ where each element is an integer denoting the qualitative state of the feature (e.g., 0 = "low", 1 = "medium", 2 = "high").

Definition 3 (Indiscernibility Relation). Based on $B^*$, the indiscernibility relation $\mathrm{IND}(A)$ defines the condition of logical identity between two objects:

$$ \mathrm{IND}(A) = \{(x_i, x_j) \in U \times U : \forall a_k \in A,\ b^*_{ik} = b^*_{jk}\} \tag{7} $$

This relation partitions the universe into equivalence classes (information granules) $U/\mathrm{IND}(A) = \{E_1, E_2, \ldots, E_L\}$, where $L \leq n$. Each granule $E_p$ groups objects sharing identical discretized attribute signatures $S_p = \langle b^*_{p1}, b^*_{p2}, \ldots, b^*_{pm} \rangle$. A granule is classified as:

* Deterministic (pure): if all objects in $E_p$ belong to the same class ($|\partial(E_p)| = 1$, where $\partial(E_p)$ is the set of distinct class values within $E_p$).
* Contradictory: if objects in $E_p$ belong to multiple classes ($|\partial(E_p)| > 1$), indicating class overlap in the discretized feature space.

### 3.4 Stage 3: Formation of Deterministic Production Rules

Definition 4 (Lower Approximation). For each decision class $X_j = \{x \in U : f(x, d) = y_j\}$, the lower approximation is defined as [12]:

$$ \underline{A}X_j = \{x \in U : [x]_A \subseteq X_j\} \tag{8} $$

where $[x]_A$ denotes the equivalence class containing $x$.

An object $x$ belongs to the lower approximation of class $X_j$ if and only if its entire equivalence class consists exclusively of objects of that class. This guarantees deterministic classification with 100% confidence.

Each deterministic granule $E_p \subseteq \underline{A}X_j$ generates a production rule:

$$ R_p: \text{IF } (a_1 \in I_{1,v_1}) \wedge \ldots \wedge (a_m \in I_{m,v_m}) \Rightarrow d = y_j \tag{9} $$

where $I_{k,v_k}$ is the value interval of the $k$-th attribute corresponding to discrete code $v_k$.

The support of rule $R_p$ quantifies its statistical significance:

$$ \mathrm{Supp}(R_p) = \frac{|E_p \cap X_j|}{|U|} \tag{10} $$

The overall quality of approximation is measured by Pawlak's Classification Quality Coefficient [8]:

$$ \gamma_A(d) = \frac{\sum_{j=1}^{K} |\underline{A}X_j|}{|U|} \tag{11} $$

which represents the proportion of objects that can be unambiguously classified using deterministic rules.

### 3.5 Stage 4: Probabilistic Rule Synthesis from Boundary Regions

Contradictory granules, i.e., those where $|\partial(E_p)| > 1$, form the boundary region $BN_A(X_j)$. For each such granule, a probabilistic rule is constructed:

$$ R_p^{\text{prob}}: \text{IF } \text{conditions} \Rightarrow d = y_j \text{ with } \mathrm{Conf}(R_p) = \frac{|E_p \cap X_j|}{|E_p|} \tag{12} $$

The confidence $\mathrm{Conf}(R_p) < 1.0$ reflects the degree of class ambiguity within the granule. Rules with confidence below a threshold may be flagged for human review in safety-critical applications. Figure 3 illustrates the relationship between lower approximation, boundary region, and upper approximation in the context of rough set classification.

![RST approximations. Deterministic granules (fully inside a target class Xj) form the lower approximation (solid green); granules partially overlapping class boundaries form the boundary region (striped yellow); their union constitutes the upper approximation. The lower approximation guarantees 100% classification confidence.](figs/fig_rough_set_approx.png)

### 3.6 Stage 5: Feature Space Minimization via Reduct Search

#### Discernibility Matrix

To identify the minimal set of features necessary for classification, the discernibility matrix $M$ [16] is constructed. For each pair of granules $(E_i, E_j)$ from different classes:

$$ c_{ij} = \{a_k \in A : b^*_{ik} \neq b^*_{jk}\}, \quad \text{when } y_i \neq y_j \tag{13} $$

The discernibility function is the conjunction of all disjunctions from non-empty cells:

$$ f_{\text{DS}}(a_1, \ldots, a_m) = \bigwedge \left\{ \bigvee(c_{ij}) : c_{ij} \neq \emptyset \right\} \tag{14} $$

#### Modified Johnson's Algorithm with Information Weights

Finding all reducts is NP-hard [16]. A weighted heuristic based on Johnson's greedy algorithm is proposed, where each attribute's selection priority incorporates its information capacity weight from the discretization stage:

$$ W(a_k) = \text{Count}(a_k \in c_{ij}) \cdot (1 - \bar{E}(a_k)) \tag{15} $$

where $\bar{E}(a_k)$ is the normalized mean conditional entropy of attribute $a_k$, computed across its discretization cut points. Attributes with lower entropy (better class separation) receive higher weights, biasing the greedy search toward features that are both discriminative and informative. The core of the attribute set is the intersection of all reducts:

$$ \mathrm{CORE}(A) = \{a_k \in A : \exists c_{ij} = \{a_k\}\} \tag{16} $$

Core attributes appear as singletons in the discernibility matrix and are indispensable for classification.

### 3.7 Stage 6: PKB Assembly

The PKB is assembled as a self-contained data structure comprising:
* Discretization thresholds $\{T^*_j\}$ for all attributes.
* The selected reduct $\mathrm{RED} \subseteq A$.
* Deterministic rules from the lower approximation (confidence = 1.0).
* Probabilistic rules from the boundary region (confidence $< 1.0$).
* Metadata: support, confidence, coverage for each rule.

This structure enables deployment without external dependencies—the PKB is a complete, inspectable artifact that can be serialized as JSON and embedded in edge devices or SCADA controllers.

### 3.8 Stage 7: Weighted Inference Engine

The inference procedure for classifying a new object $X_{\text{new}} = \langle b_1, \ldots, b_m \rangle$ proceeds as follows:

Step 1: Discretization. Apply the stored quantization functions:

$$ x^*_j = \phi_j(b_j), \quad \forall j \in \{1, \ldots, m\} \tag{17} $$

Step 2: Rule Activation. Search the knowledge base for all rules whose conditions match the discretized signature, using only attributes in $\mathrm{RED}$.

Step 3: Conflict Resolution. When multiple rules for different classes are activated, compute the integral class support score:

$$ S(y_k) = \sum_{\substack{R_i \in R_{\text{active}} \\ \text{Class}=y_k}} \mathrm{Conf}(R_i) \cdot \mathrm{Supp}(R_i) \cdot w_i \tag{18} $$

where $w_i$ is the interval reliability weight derived from the density profile at the discretization zone. The final classification is:

$$ y^* = \arg\max_{y_k} S(y_k) \tag{19} $$

Step 4: Soft Matching Fallback. If no rule matches exactly, a Hamming distance-based soft matching procedure finds the closest rule signature. If no acceptable match is found, the default class (most frequent in $Y$) is assigned.

## 4. Case Study

### 4.1 Experimental Setup

#### SCADA Simulation Dataset

To validate the PKB methodology in the context of PV fire hazard monitoring, a simulated SCADA dataset was constructed with the following characteristics:

* Size: $n = 1000$ samples (PV module row measurements).
* Features: 3 continuous sensor measurements:
    * Thermal anomaly intensity ($X_i$): range [0, 100], threshold 50.
    * Bypass diode temperature ($X_{1i}$): range [20, 120]$^\circ$C, threshold 70$^\circ$C.
    * Fire sensor signal ($X_{2i}$): range [0, 10], threshold 5.
* Classes: 5 operational states based on Boolean logic combinations (Table 1).
* Noise: Gaussian noise added to class-conditional distributions (random seed 42).

Table 1: SCADA operational states and their Boolean logic definitions, corresponding to the fire hazard truth table implemented in the PV monitoring SCADA system.

| State | $X_i$ | $X_{1i}$ | $X_{2i}$ | Count |
| :--- | :---: | :---: | :---: | :---: |
| Normal | Low | Low | Low | 300 |
| Safe Bypass | High | High | Low | 250 |
| Fire Hazard | High | Low | Low | 150 |
| Faulty Diode | Low | High | Low | 150 |
| Active Fire | Any | Any | High | 150 |

#### Benchmark Datasets

Two publicly available benchmark datasets from the UCI Machine Learning Repository [30] were used:
* Iris [31]: 150 samples, 4 features, 3 classes (setosa, versicolor, virginica).
* Wine [32]: 178 samples, 13 features, 3 classes (cultivar types).

#### Evaluation Protocol

For the SCADA dataset, the full dataset was used for both training and validation to assess the PKB's capacity to model the complete feature space. For benchmark datasets, 5-fold stratified cross-validation [33] was employed with a fixed random seed of 42 for reproducibility. The PKB methodology was compared against two baselines: Decision Tree (CART implementation from scikit-learn [30]) and Gaussian Naive Bayes.

### 4.2 WEDD Discretization Results

The WEDD algorithm with $\alpha = 0.6$ produced the following discretization thresholds (Table 2).

Table 2: WEDD discretization thresholds compared to ground-truth class boundaries. The algorithm successfully recovers thresholds close to the true values, with cut points positioned at density valleys.

| Feature | Truth | WEDD Cuts | Bins |
| :--- | :---: | :--- | :---: |
| Thermal intensity | 50.0 | 42.8, 47.5, 54.7 | 4 |
| Diode temp. ($^\circ$C) | 70.0 | 58.5, 70.0, 76.0 | 4 |
| Fire signal | 5.0 | 5.4 | 2 |

Figure 4 presents the visualization of the WEDD discretization results, showing class-conditional histograms, KDE density curves, and the selected cut points for each feature. Key observations include: (1) the bypass diode temperature cut at 70.0$^\circ$C exactly recovers the ground truth threshold; (2) thermal anomaly intensity receives three cuts that bracket the true boundary at 50.0 from both sides, creating a transition zone that captures the class overlap; (3) fire sensor signal requires only one cut at 5.4, positioned in the density valley between fire and non-fire populations.

![WEDD discretization results for three SCADA sensor features. Each panel shows: class-conditional histograms (colored bars), Gaussian KDE density curves (solid lines), and selected cut points (vertical dashed lines). The bypass diode temperature threshold at 70.0 degrees Celsius nearly exactly recovers the ground truth. Fire sensor signal requires only a single cut point at 5.4, closely matching the true threshold of 5.0. The multi-criteria objective places cuts at density valleys while maximizing class separation.](figs/fig_discretization.png)

### 4.3 Granulation and Rule Extraction Results

The discretized feature space ($4 \times 4 \times 2 = 32$ combinations) produced 32 information granules, achieving a 31$\times$ compression ratio from 1000 objects. Table 3 summarizes the granulation results. The 19 deterministic rules are distributed as follows: 16 rules for Active Fire (covering 98.0% of the class), 1 each for Faulty Diode, Normal, and Safe Bypass. Notably, the Fire Hazard class has zero deterministic rules: it resides entirely in the boundary region, reflecting the inherent difficulty of distinguishing fire hazard conditions (high thermal anomaly + low diode temperature) from adjacent states. The 13 probabilistic rules from contradictory granules complete the coverage, with confidence ranging from 50.0% (P12, equal Fire Hazard/Normal split) to 99.6% (P5, near-deterministic Normal classification).

Table 3: Information granulation results from IND($A$) equivalence class construction.

| Metric | Value |
| :--- | :---: |
| Total objects ($|U|$) | 1,000 |
| Total granules ($L$) | 32 |
| Compression ratio ($L/|U|$) | 0.032 |
| Deterministic granules | 19 (59.4%) |
| Contradictory granules | 13 (40.6%) |
| $\gamma_A(d)$ (Classification quality) | 0.150 |
| Objects in lower approximation | 150 (15.0%) |
| Objects in boundary region | 850 (85.0%) |
| Deterministic rules extracted | 19 |
| Probabilistic rules extracted | 13 |
| Total production rules | 32 |

### 4.4 Reduct Analysis Results

The discernibility matrix analysis revealed 351 non-empty entries from 496 granule pairs with different majority classes. All three attributes appear as singletons in the discernibility matrix (Table 4). Since all three attributes have singleton entries, $\mathrm{CORE}(A) = \mathrm{RED} = A$ (the full attribute set). No feature reduction is possible, i.e., each sensor captures a distinct physical phenomenon that is irreplaceable for classification. This confirms the well-designed nature of the SCADA sensor suite.

Table 4: Reduct analysis: feature weights and discernibility singletons.

| Attribute | Weight | Singletons | Score |
| :--- | :---: | :---: | :---: |
| Thermal intensity | 0.328 | 15 | 88.5 |
| Diode temperature | 0.314 | 17 | 20.4 |
| Fire signal | 0.280 | 16 | 4.5 |

### 4.5 Inference Validation Results

The PKB inference engine achieved the results shown in Table 5 and Figure 5. The critical safety class Fire Hazard achieves 98.0% recall (147 of 150 correctly identified) despite relying entirely on probabilistic rules. The precision of 90.7% indicates some false alarms from the Normal class, which is an acceptable trade-off in safety-critical applications where missed hazards carry catastrophic consequences.

Table 5: PKB inference engine validation metrics on the SCADA dataset ($n = 1000$). The system achieves 96.2% overall accuracy with perfect deterministic rule performance.

| Class | Prec. | Rec. | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
| Active Fire | 1.000 | 0.980 | 0.990 | 150 |
| Faulty Diode | 0.958 | 0.913 | 0.935 | 150 |
| Fire Hazard | 0.907 | 0.980 | 0.942 | 150 |
| Normal | 0.983 | 0.957 | 0.970 | 300 |
| Safe Bypass | 0.953 | 0.976 | 0.964 | 250 |
| Macro avg. | 0.960 | 0.961 | 0.960 | 1000 |
| Weighted avg. | 0.963 | 0.962 | 0.962 | 1000 |

![Confusion matrix for PKB inference validation on the SCADA dataset. The diagonal dominance confirms high classification accuracy across all five operational states. The primary error sources are misclassifications between Faulty Diode and Safe Bypass (10 cases) and between Normal and Fire Hazard (9 cases), both occurring in boundary regions where sensor signatures overlap.](figs/fig_confusion_matrix.png)

Analysis by rule type reveals a clear dichotomy: deterministic rules achieve 100% accuracy (150/150), while probabilistic rules achieve 95.5% accuracy (812/850). All 38 misclassifications originate from probabilistic rules in high-conflict boundary granules.

### 4.6 Benchmark Comparison Results

Table 6 and Figure 6 present the comparative results on the Iris and Wine benchmark datasets. On the Iris dataset (4 features, 3 classes), the PKB achieves 93.3% accuracy, within 2.0 percentage points of the best baseline (Decision Tree at 95.3%). On the Wine dataset (13 features, 3 classes), the PKB achieves 87.6% accuracy with greedy reduct-based feature selection reducing the feature space from 13 to 3–6 features per fold. While Naive Bayes excels on Wine (97.2%) due to the near-Gaussian feature distributions satisfying its independence assumption, the PKB significantly outperforms on interpretability.

Table 6: Benchmark comparison using 5-fold stratified cross-validation. Mean $\pm$ standard deviation reported. Best results in bold.

| Dataset | Method | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| Iris | PKB | $0.933 \pm 0.030$ | $0.933 \pm 0.030$ |
| | Decision Tree | $\mathbf{0.953 \pm 0.034}$ | $\mathbf{0.953 \pm 0.034}$ |
| | Naive Bayes | $0.947 \pm 0.040$ | $0.947 \pm 0.040$ |
| Wine | PKB | $0.876 \pm 0.078$ | $0.873 \pm 0.083$ |
| | Decision Tree | $0.893 \pm 0.038$ | $0.896 \pm 0.036$ |
| | Naive Bayes | $\mathbf{0.972 \pm 0.025}$ | $\mathbf{0.973 \pm 0.024}$ |

![Benchmark comparison of PKB methodology against Decision Tree and Naive Bayes baselines on Iris and Wine datasets using 5-fold stratified cross-validation. The PKB achieves competitive accuracy while providing the unique advantage of interpretable IF-THEN production rules with explicit confidence scores. Error bars represent +/- 1 standard deviation across folds.](figs/fig_benchmark_comparison.png)

### 4.7 Application Context: PV Fire Hazard Detection

Figures 7 and 8 show the detection accuracy improvements and system efficiency gains, respectively. The PKB methodology integrates with the SCADA Boolean logic layer (Figure 9) by replacing the binary thresholding with interpretable production rules that provide confidence-weighted classification, enabling operators to understand the reasoning behind each fire hazard assessment. Figure 10 demonstrates the system's ability to discriminate between different defect types based on thermal profile characteristics.

![Detection accuracy improvements in the PV monitoring CPS. The ensemble method combining M2 (large defect detection) and M3 (hot spot detection) thermal palettes achieves mAP@0.5 of 0.96 for Class 1 defects (+3%), 0.90 for Class 2, and 0.95 for Class 3 (+2%). The PKB methodology complements this detection layer by providing interpretable classification of operational states from the detected thermal anomalies.](figs/fig_accuracy_comparison.png)

![System efficiency improvements in the edge-cloud CPS architecture. Left: 31.7% reduction in data transmission volume. Center: 32.5% reduction in processing time through edge computing. Right: 23–28% cumulative reduction in false positives. The PKB inference engine adds interpretable fire hazard classification on top of these efficiency gains, enabling proactive safety assessment via SCADA Boolean logic.](figs/fig_system_efficiency.png)

![Fire hazard decision logic implemented in the SCADA system. The Boolean truth table correlates three sensor inputs (Xi, X1i, X2i) to classify operational states. The PKB methodology enhances this logic by providing confidence-weighted rules instead of hard binary thresholds, enabling more nuanced assessment of borderline cases.](figs/fig_hazard_logic.png)

![Thermal profiles for two critical defect types in PV modules. Top: Hot Spot defect showing sharp Gaussian-like temperature peak (detected by M3 palette). Bottom: Bypass Diode Activation showing step-function profile (detected by M2 palette). These thermal signatures, when quantified by SCADA sensors, serve as input features for PKB classification.](figs/fig_thermal_profiles.png)

## 5. Discussion

The results presented in this study validate the efficacy of the PKB methodology as a robust alternative to opaque black-box classifiers for safety-critical industrial applications. As highlighted in Section 1, a primary challenge in modern CPS is bridging the gap between the high detection accuracy of deep learning models, such as YOLO architectures [1], [5], and the interpretability required for operator trust. The proposed framework successfully addresses this by converting continuous sensor data into symbolic knowledge. A critical factor in this success is the WEDD method. Unlike standard entropy-based discretization which focuses solely on class separation, often splitting natural clusters, the proposed approach incorporates KDE to locate thresholds in the density valleys between subpopulations. As visualized in Figure 4, this multi-criteria optimization allowed the system to recover ground-truth physical thresholds with remarkable precision, such as the 70.0$^\circ$C cut point for bypass diode temperature, without requiring manual calibration. This confirms that augmenting information entropy with density constraints preserves the semantic structure of the data, creating a stable foundation for rule induction.

The application of RST provided a rigorous mechanism for quantifying uncertainty, a feature often absent in standard decision trees or rule induction algorithms like CN2 or RIPPER. The granulation results in Table 3 reveal that while only 15% of the simulated SCADA objects fell into the lower approximation (deterministic zone), the system maintained high accuracy in the boundary region through confidence-weighted inference. This distinction is operationally significant: the system can autonomously execute actions based on deterministic rules (confidence = 1.0) while flagging probabilistic rules (confidence $< 1.0$) for human verification. This capability supports the hybrid decision-making model described in recent literature [7], where automated efficiency must be balanced with human oversight. Furthermore, the exact 100% accuracy achieved by deterministic rules (Table 5) empirically validates the theoretical property of the lower approximation: when the data structure is unambiguous, the derived logical rules are infallible.

Performance analysis on the SCADA dataset demonstrates that the methodology meets the stringent requirements of fire safety monitoring. The confusion matrix in Figure 5 indicates that the system is particularly effective at prioritizing recall for the most critical state. Although the Fire Hazard class lacked deterministic rules due to significant feature overlap, the probabilistic inference engine achieved 98.0% recall, missing only 2% of hazards. The trade-off is a precision of 90.7%, implying a moderate false alarm rate. In the context of PV plant protection [2], this bias is desirable; the economic and safety cost of a missed fire far outweighs the operational cost of investigating a false positive. Comparative benchmarking on the Iris and Wine datasets (Table 6 and Figure 6) places the proposed approach within competitive range of established algorithms. While Gaussian Naive Bayes outperformed the proposed method on the Wine dataset (97.2% vs. 87.6%) due to the specific distributional properties of that dataset, the approach presented here offers the unique advantage of generating explicit IF-THEN rules, as shown in Section 4.5, which are legally auditable and directly understandable by domain experts.

The architectural integration of this methodology into edge-cloud environments offers distinct advantages over purely cloud-based or purely neural-based solutions. As shown in Figure 8, the system contributes to a significant reduction in data transmission and processing latency. By serializing the entire knowledge base—including discretization thresholds, reducts, and rule sets—into a lightweight JSON format of approximately 32 KB, the system is highly portable. It can be embedded directly into the control logic of edge devices like the NVIDIA Jetson AGX Orin or even lower-power microcontrollers, functioning as an interpretable logic layer between the neural detection front-end and the SCADA actuation back-end. This aligns with the industry trend toward edge intelligence [4], enabling decentralized decision-making that remains functional even during network interruptions.

Despite these strengths, several limitations and open research challenges remain. First, the current validation relies on simulated SCADA data with Gaussian noise. Real-world sensor data from PV plants often exhibit non-Gaussian noise, temporal drift, and sensor faults that may degrade the performance of the density estimation component. The relatively low classification quality coefficient ($\gamma_A(d) = 0.15$) suggests that the three-feature space contains substantial overlap, which might be irreducible without additional sensor modalities. Furthermore, the feature selection mechanism relies on a modified greedy algorithm. While the information-capacity weighting improves upon the standard Johnson's algorithm, it does not guarantee a globally optimal reduct, potentially leading to rule sets that are larger than necessary. The performance drop on the high-dimensional Wine dataset suggests that the current heuristic may struggle with feature interactions in larger spaces. Future work must address these challenges by exploring evolutionary optimization for reduct search and validating the methodology on diverse, real-world operational datasets to ensure robustness against environmental variability and sensor degradation.

## 6. Conclusion

This study successfully establishes the PKB approach as a comprehensive and rigorous framework for transforming continuous cyber-physical sensor data into transparent, actionable intelligence. By synthesizing a novel WEDD technique with the mathematical foundations of RST, a pipeline has been developed that generates production rules that are not only accurate but also linguistically interpretable and logically sound. The methodology was rigorously validated through a seven-stage processing pipeline, demonstrating its capability to handle the complexities of industrial monitoring. On the simulated SCADA fire hazard dataset, the system achieved a compelling 96.2% overall accuracy, with the lower approximation yielding perfect 100% accuracy for deterministic cases. Crucially for safety-critical infrastructure, the system achieved a 98.0% recall rate for fire hazards, ensuring that dangerous conditions are reliably detected even when feature signatures partially overlap with normal operational states. The benchmarking results on the Iris (93.3%) and Wine (87.6%) datasets confirm that while the approach is tailored for industrial logic, it maintains competitive performance across general classification tasks compared to black-box baselines. The primary limitation identified lies in the greedy nature of the feature reduction stage, which may yield sub-optimal subsets in high-dimensional spaces, and the reliance on simulated noise models.

Looking forward, three key research directions are envisioned to enhance this framework: the integration of palette-invariant radiometric models to improve input robustness against environmental thermal noise, the application of evolutionary algorithms to optimize the trade-off between rule set conciseness and classification coverage, and the development of online learning mechanisms that allow the knowledge base to adapt dynamically to the aging of PV components.

## References

1[Xie2024]
2[Lysyi2025Enhanced]
3[Thakfan2024]
4[Svystun2024]
5[Terven2023]
6[Melnychenko2024]
7[Liu2026]
8[Pawlak1991]
9[Quinlan1986]
10. [Quinlan1993]
11. [Breiman1984]
12. [Pawlak1982]
13. [Tetteh2025]
14. [Slowinski2023]
15. [Toulabinejad2024]
16. [Skowron1992]
17. [Slowinski1992]
18. [Greco2001]
19. [Pawlak1998]
20. [Grzymala1992]
21. [Garcia2013]
22. [Dougherty1995]
23. [Liu2002]
24. [Xun2021]
25. [Rosenblatt1956]
26. [Parzen1962]
27. [Silverman1986]
28. [Lysyi2025Method]
29. [Shannon1948]
30. [Pedregosa2011]
31. [Fisher1936]
32. [Aeberhard1994]
33. [Kohavi1995]
