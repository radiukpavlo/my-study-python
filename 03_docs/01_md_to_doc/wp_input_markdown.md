# Architecture of Cyberphysical Systems for UAV-Based Late-Fusion Defect Detection and Fire-Risk Mitigation in Photovoltaic Modules

**Anatoliy Sachenko** [2], **Pavlo Radiuk** [1]*, **Mykola Lysyi** [1], **Oleksandr Melnychenko** [1]

* [1] Khmelnytskyi National University, 11, Instytutska Str., Khmelnytskyi, 29016, Ukraine
* [2] West Ukrainian National University, 11, Lvivska Str., Ternopil, 46009, Ukraine

\* *Corresponding author*

---

## Abstract

The **Subject** of this research focuses on the architectural advancement of inspection frameworks for large-scale photovoltaic (PV) power plants. As PV infrastructure expands globally, the reliance on manual analysis or offline, batch-processed aerial thermography creates significant bottlenecks. Current unmanned aerial vehicle (UAV) workflows often generate massive datasets that are weakly geo-referenced and disconnected from real-time plant supervision, leading to delayed maintenance and undetected safety hazards such as fire risks caused by bypass diode failures. The **goal** of this study is to overcome these operational disconnects by designing, developing, and validating a novel cyber-physical system (CPS) architecture. This architecture aims to transform UAV-based inspection from a passive data collection task into an active, closed-loop diagnostic service. The specific objectives include implementing real-time on-board defect segmentation, ensuring precise module-level traceability through edge-computing geo-indexing, and integrating detection outputs directly with SCADA systems to trigger immediate, safety-critical alarms based on formal logic. The **Methods** employed involve a multi-tiered distributed computing framework comprising a UAV tier, an on-site edge tier, and a cloud tier. The UAV captures synchronized RGB and radiometric thermal imagery, processing it on-board using a lightweight YOLOv11 segmentation model. To address the domain shift caused by thermal visualization techniques, the study introduces a late-fusion ensemble strategy that combines detections from two distinct thermogram representations: a wide-range two-color palette (M2) and a high-contrast three-color palette (M3). This is complemented by a "size-based routing" mechanism to optimize sensitivity for both subtle cracks and gross delamination. At the edge tier, raw detections are refined using RTK-assisted spatial clustering and de-duplication algorithms to prevent over-counting. Furthermore, a deterministic Boolean logic layer correlates visual defects with bypass diode states to assess fire risks explicitly. The **Results**, reported as mean values under stratified five-fold cross-validation, indicate that the proposed CPS architecture significantly outperforms single-modality baselines. The on-board YOLOv11 model achieved macro mAP@0.5 scores of 0.91 and 0.90 for M2 and M3 palettes, respectively. The late-fusion ensemble proved superior, elevating crack detection mAP@0.5 from 0.93 to 0.96 and enhancing delamination detection to 0.95 mAP@0.5. Crucially, the pipelined architecture reduced the end-to-end per-frame processing latency from 4.235 s to 2.858 s, facilitating near-real-time responsiveness. Field validation conducted on two PV installations showed a Root Mean Square Error of 0.71 defects per inspected PV string against manual expert counts. Sensitivity analysis highlighted that a flight altitude of 10 m provides the optimal balance, yielding 93% precision and 90% recall, whereas increasing altitude to 15 m caused a sharp drop in precision to 84%. In **conclusion**, this research demonstrates that treating UAV inspection as an integrated cyber-physical service rather than a standalone sensing task fundamentally improves defect traceability and operational utility, offering a scalable solution for preventive maintenance and automated fire-risk mitigation in the renewable energy sector.

**Keywords:** cyber-physical systems, photovoltaic modules, unmanned aerial vehicles, infrared thermography, edge computing, deep learning segmentation, SCADA integration

---

## 1. Introduction

Photovoltaic (PV) energy has moved from a niche technology to a critical component of modern power systems. At the same time, PV plants remain exposed to harsh outdoor environments and to a long list of failure modes that are mundane in origin but costly in aggregate [1]. Microcracks can be introduced during manufacturing, transportation, installation, or hail events; they may evolve into inactive cell regions, hot spots, and power losses [2]. Encapsulant aging and delamination can trap moisture and accelerate corrosion. Soiling reduces irradiance on the cells and changes the thermal regime, and damaged interconnects or bypass diodes can create abnormal heating that threatens safety. Reliability-centered maintenance is therefore not optional [3]: predictive and preventive inspection directly affects energy yield and plant lifetime.

Infrared (IR) thermography is one of the most informative non-contact diagnostic modalities for PV systems. It provides spatial maps of temperature that reflect electrical mismatch, increased series resistance, shunting, and local degradation. Reviews show that thermography can reveal a broad spectrum of PV anomalies but also highlight that its diagnostic value depends on acquisition conditions, emissivity assumptions, viewing geometry, and the ability to distinguish defect signatures from reflections and environmental gradients [4], [5]. Comparative studies further demonstrate that defects may manifest differently under illumination and in dark conditions, which means that the inspection protocol is as important as the camera itself [6].

Unmanned aerial vehicles (UAVs) have recently become the default platform for large-scale PV thermography. A UAV can cover utility-scale fields in minutes and can inspect rooftop installations that are difficult to access [7], [8]. Multiple studies report the operational benefits of UAV thermography, including improved coverage and reduced inspection time [9], but they also document practical constraints: flight altitude and speed control spatial resolution and motion blur; time of day and weather control thermal contrast; and specular reflections from the glass surface can produce misleading hot regions [10], [11]. These constraints are not merely nuisances; they are the reason why "laboratory-grade" image analysis pipelines often fail when deployed in the field.

Automation is the second major trend. Deep learning has achieved strong results in PV defect classification and localization across IR and RGB imagery, and surveys describe a rapidly growing set of architectures ranging from classical convolutional-based classifiers to modern one-stage detectors and segmentation networks [12], [13]. Recent UAV-oriented approaches use variants of the YOLO family to detect modules and anomalies in thermal and visible data [14], [15], and methods based on thermal video show that automation can be extended from single frames to temporal streams [16]. However, a recurring limitation is that many published solutions implicitly assume an offline workflow: data are collected, uploaded, and processed in a batch manner [12], [17]. In operational PV plants, the inspection process is a closed-loop service that must satisfy latency, bandwidth, and reliability requirements, and it must deliver outputs that are traceable to physical modules and actionable by operators [18].

This observation motivates a systems-level framing. A PV inspection pipeline is not only a computer vision model; it is a cyber-physical system (CPS) [19] in which sensing, computation, communication, and control are coupled. A UAV captures physical measurements, but the plant requires cyber actions: alarms, maintenance tickets, and safety procedures. Edge computing offers a natural architectural answer: it can reduce bandwidth by processing data near the source and can provide fast responses without depending on cloud connectivity [20]. Yet, despite the maturity of edge computing in industrial monitoring, UAV inspection solutions for PV plants rarely provide an explicit CPS architecture that integrates on-board inference, edge aggregation, long-term cloud analytics, and SCADA-aware decision logic [21]. Similarly, the literature on PV thermography tends to emphasize detection accuracy while under-reporting the geo-referencing and de-duplication steps required to avoid inflated counts and alarm fatigue.

The problem under consideration in this paper is therefore the design and validation of a CPS architecture for UAV-based PV defect monitoring that is accurate and operationally actionable. The technical gap is not simply "detect more defects" but "detect defects as part of a robust service" with consistent geo-indexing, real-time execution on embedded hardware, and integration with plant supervision and safety monitoring. This is especially relevant for fire-risk mitigation, where abnormal heating and bypass diode failures can turn local defects into hazardous events. Integrating UAV-based detection with SCADA logic aligns with recent work that treats UAV and AI as components of plant safety monitoring pipelines [22], [23].

The goal of this study is to improve the operational usefulness of UAV-based PV module inspection by developing a cyber-physical architecture that integrates on-board deep learning, edge computing, cloud analytics, and SCADA-aware decision logic into a single coordinated workflow. To achieve this goal, this study introduces three main scientific contributions:
*   A multi-tier CPS architecture (UAV–edge–cloud) for PV inspection that defines data flow, responsibilities, and decision points, including RTK-assisted geo-indexing and de-duplication to produce traceable, module-level defect events.
*   A thermography-oriented on-board inference pipeline based on YOLOv11 segmentation with palette-aware post-processing and late-fusion ensemble across two thermogram representations, achieving strong detection quality while reducing end-to-end processing time.
*   An integration of detection outputs with an interpretable SCADA-aware logic layer for hazard inference.

## 2. Related Works

Research relevant to UAV-based PV defect monitoring spans sensing physics, computer vision, and systems engineering. This section briefly positions the present work within that landscape and clarifies how the proposed approach to cyber-physical operation differs from detector-only contributions.

Thermography is widely used to detect hot spots, mismatch, and degradation patterns in PV modules. Comprehensive reviews emphasize that thermal signatures are influenced by irradiance, ambient temperature, wind, and viewing angle, and they discuss recommended inspection procedures and limitations [4], [5]. Gallardo-Saavedra et al. [6] compared illumination and dark conditions and showed that defect detectability depends on the operating state, which reinforces the need for consistent flight protocols and for contextual information such as irradiance and ambient conditions. Practical UAV deployments have been evaluated in field studies that highlight how flight altitude and motion control affect spatial resolution and therefore anomaly separability [7], [11]. These findings motivate the emphasis on flight-parameter sensitivity and reflection-aware planning.

Deep learning for PV diagnostics has progressed from classification of cropped module images to detection and segmentation at the plant scale. Surveys summarize the expanding use of CNNs, object detectors, and hybrid models in PV monitoring [17], [24]. Early work by Dotenco et al. [25] demonstrated automatic PV module detection and defect analysis in aerial infrared imagery using statistical tests and classical vision, reporting strong defect classification performance while also emphasizing the importance of robust pre-processing. Later work increasingly relies on one-stage detectors. Di Tommaso et al. [14] proposed a multi-stage pipeline using YOLOv3 for module and anomaly detection in both IR and visible imagery, which improved automation but still required careful separation of tasks and datasets. Xie et al. [18] introduced ST-YOLO for PV fault detection from IR images and reported very high mAP@0.5 under their experimental conditions. Other studies focus on thermographic feature learning and robustness, including deep learning strategies for fault diagnosis from thermograms [26] and segmentation-based techniques to delineate modules and defects [27]. These works motivate the use of a segmentation-capable detector (YOLOv11-seg) and the attention to palette-induced domain shift.

While thermography is central, many deployments use RGB imagery to provide context and reduce false alarms. RGB-based deep learning has been used to classify failure signatures and operational issues in PV plants [28]. Multi-modal pipelines can also support human interpretability, because operators can inspect RGB context when thermal signatures are ambiguous [29]. The proposed architecture therefore assumes synchronized RGB and thermal acquisition, even though the defect detector in this paper focuses on thermal masks.

From a systems perspective, UAV inspection is an instance of distributed sensing with stringent constraints [30]. Edge computing is often advocated for industrial IoT because it reduces cloud dependence and provides low-latency decisions [20]. Tang et al. [22] demonstrated an edge–cloud deep learning architecture for detecting linear defects in large-scale PV plants, showing that distributed computation can support plant-scale monitoring and reduce data transfer. However, many edge–cloud works stop at "detection at the edge" and do not explicitly model the feedback loop to plant operations, as pointed out by Ferlito et al. [23]. This study extends this direction by adding RTK-based geo-indexing, duplicate suppression, and SCADA-aware fire-risk logic, motivated by the observation that PV defects have both performance and safety implications [21].

Thus, the objective of this study is to develop and validate an end-to-end CPS architecture for UAV-based PV defect monitoring that is accurate, geo-referenced, and actionable. To fulfill this objective, three tasks must be completed: (1) formalize the UAV–edge–cloud architecture and define interfaces for data, geo-tags, and alarms; (2) develop and validate an on-board thermography detection pipeline with palette-aware post-processing and fusion; and (3) perform quantitative evaluation and field validation, including sensitivity analysis to operational conditions, to assess whether the architecture improves real inspection workflows rather than only offline metrics.

## 3. Methods and Materials

This section details the proposed approach. It describes the CPS architecture, sensing and communication design, reflection-aware flight planning, dataset construction, on-board deep learning inference, edge/cloud analytics, and the SCADA-aware decision layer. The guiding principle is to treat defect detection as a cyber-physical workflow: sensing, computation, communication, and action must be co-designed.

### 3.1. System Architecture and Operational Loop

![Proposed architecture of cyber-physical systems for UAV-based PV defect monitoring. The UAV tier performs synchronized sensing and on-board YOLOv11 segmentation. The edge tier aggregates detections, performs RTK-based geo-indexing and de-duplication, and interfaces with SCADA. The cloud tier provides long-term storage, analytics, and model lifecycle management.](figs/76_fig_1_v1.jpg)
*Figure 1: Proposed architecture of cyber-physical systems for UAV-based PV defect monitoring.*

This study operationalizes PV inspection as a closed loop that starts with sensing and ends with a plant-level action. The CPS is organized into three computational tiers (Figure 1): the UAV tier, the edge tier deployed at the PV plant, and the cloud tier for long-term analytics. An operator layer interacts with the CPS through a dashboard and through SCADA/HMI interfaces.

The UAV tier is responsible for time-critical tasks. It captures synchronized RGB and thermal frames and performs on-board inference to produce defect candidates. The edge tier is responsible for plant-context tasks: it performs RTK-assisted geo-indexing, de-duplicates repeated detections, aggregates them into module-level events, and interfaces with SCADA for alarms and maintenance triggers. The cloud stores mission artifacts and provides dashboards for trend analysis, module history, and inspection reporting.

Let a mission produce a time-ordered stream of frames $\{I_t\}_{t=1}^{T}$, where each frame contains an RGB image $I_t^{\mathrm{rgb}}$, a thermal image $I_t^{\mathrm{th}}$, and a navigation record $n_t$ (RTK position, altitude, yaw, and timestamp). On-board inference produces a set of detections:

$$
\mathcal{D}_t = \{(c_k, M_k, s_k, g_k)\}_{k=1}^{K_t},
$$

where $c_k$ is the defect class (crack, soiling, delamination), $M_k$ is a segmentation mask, $s_k$ is a confidence score, and $g_k$ is a geo-tag (estimated module location in plant coordinates).

The edge server receives $\mathcal{D}_t$ and performs (i) spatial clustering and de-duplication across time, (ii) association of detections with module identifiers and strings, and (iii) state inference that maps detections and SCADA signals to operational actions. The cloud stores mission artifacts and provides dashboards for trend analysis, module history, and inspection reporting.

### 3.2. UAV Platform, Sensors, and Communication

The CPS assumes a UAV platform capable of stable flight in predefined patterns (grid or corridor scanning) and equipped with a combined RGB/thermal payload. The thermal channel provides the primary signal for anomalous heating patterns; the RGB channel provides contextual information and supports operator validation. While the architecture is platform-agnostic, the reported implementation is comparable to Matrice-class UAVs with a dual-sensor gimbal payload (RGB + thermal) and RTK positioning.

From a CPS viewpoint, geo-referencing is as important as imaging. RTK positioning reduces drift and enables consistent mapping of detections to physical modules across missions. In addition, accurate timestamps allow synchronization between the thermal camera, the RGB camera, and the flight controller. The UAV logs the navigation record $n_t$ for each frame, including RTK position and altitude. This record is later used for de-duplication and for associating detections with the plant layout.

Communication is treated as a constrained resource. Streaming full-resolution thermal video is often infeasible over standard UAV links, especially in large plants. Instead, the UAV transmits compact detection messages by Equation (1) to the edge server in near-real time. Full-resolution images may be cached on-board and uploaded post-flight or selectively uploaded when the edge server requests evidence. This design follows the general rationale for edge computing in industrial monitoring: local inference reduces bandwidth, and plant-level decisions can be made without dependence on cloud connectivity [20].

### 3.3. Reflection-Aware Viewpoint Selection

UAV thermography of PV modules is vulnerable to specular reflections from the glass surface. If the camera view aligns with the specular reflection direction of the sun, the thermogram can be dominated by glare, producing false hotspots and corrupting the thermal contrast needed for defect detection. To mitigate this, a reflection-aware viewpoint selection algorithm is incorporated with RTK geometry and the solar position.

![Geometric intuition for reflection-aware viewpoint selection. Given the module plane normal $n$ and the solar ray direction $s$, the specular reflection direction $r$ is computed. The desired camera viewing direction $v^\star$ is obtained by rotating away from $r$ (Rodrigues rotation) to reduce glare while maintaining sufficient incidence for thermal contrast.](figs/76_fig_2_v1.jpg)
*Figure 2: Geometric intuition for reflection-aware viewpoint selection.*

Let three RTK-referenced points on the module plane be $p_1, p_2, p_3 \in \mathbb{R}^3$ (e.g., corners of a representative module). The module normal vector is computed as:

$$
n = \frac{(p_2-p_1)\times(p_3-p_1)}{\|(p_2-p_1)\times(p_3-p_1)\|}.
$$

Let $s$ be the unit vector pointing from the module to the sun (obtained from a solar ephemeris model at timestamp $t$). The specular reflection direction is:

$$
r = s - 2(n^\top s)\,n.
$$

If the camera viewing direction aligns with $r$, specular reflections dominate. A target viewing direction $v^\star$ is defined by rotating $r$ around the normal $n$ by an angle $\theta$ selected so that the reflection direction falls outside the camera field of view. Using Rodrigues' rotation formula, the rotated vector is:

$$
v^\star = r\cos\theta + (n\times r)\sin\theta + n(n^\top r)(1-\cos\theta).
$$

Algorithm 1 summarizes this viewpoint selection procedure. In practice, $\Theta$ can be a small set of candidate angles (e.g., $\pm 5^\circ, \pm 10^\circ$) and the glare score can be approximated by the angular distance between $v_\theta$ and $r$ relative to the camera field of view. The key point is that the algorithm is lightweight and compatible with real-time decision-making at the edge tier.

**Algorithm 1: Reflection-aware viewpoint selection for UAV thermography**
```text
Require: RTK plane points p1, p2, p3, solar direction s, candidate rotation angles Θ
Ensure: View direction v* that reduces specular glare
1: Compute surface normal n using Eq. (2)
2: Compute reflection direction r using Eq. (3)
3: for all θ in Θ do
4:     Compute candidate view v_θ using Eq. (4)
5:     Evaluate a glare score (expected reflection within camera FOV)
6: end for
7: Select v* = arg min_θ GlareScore(v_θ)
8: return v*
```

### 3.4. Thermogram Representations and Dataset Construction

Thermal cameras often support multiple color palettes that map temperature ranges to colors for visualization. While these palettes are designed for human interpretation, they also change the statistical distribution of pixel values presented to a learning model. In the conducted experiments, two thermogram representations are considered: a two-color palette (denoted **M2**) and a three-color palette (denoted **M3**). The key difference is the effective temperature range emphasized by the rendering. In the reported setup, M3 uses a narrower dynamic range of approximately 15 °C, which increases sensitivity to small temperature contrasts (on the order of 0.1–0.5 °C) but can saturate extreme anomalies. M2 uses a wider range of approximately 35 °C, which reduces sensitivity to tiny hot spots but covers the full spectrum from mild mismatch heating to severe damage. M2 and M3 are treated as two benchmark thermogram representations derived from the same physical scenes to evaluate robustness to palette-induced domain differences and to motivate selective fusion.

To develop and validate the proposed approach, two publicly available UAV thermography datasets were additionally utilized to provide complementary defect coverage:
1.  **STHS-277** [31]: This dataset contains 277 full-frame thermographic images capturing snail-trail and hotspot defects and includes environmental metadata. We extend the provided annotations by adding bounding boxes for all PV panel instances to enable end-to-end detector training.
2.  **PVF-10** [32]: This is a large-scale, high-resolution UAV thermal dataset with 5,579 annotated crops of individual solar panels from eight power plants and a fine-grained taxonomy of ten fault classes.

For the on-board segmentation experiments reported in this paper, we curated a compact three-class mask dataset (crack, soiling, delamination) by combining field thermography with selected samples from the public datasets above. Images were annotated using the Computer Vision Annotation Tool (CVAT) [33] and exported as polygon masks suitable for segmentation training.

Because the minority classes are small, evaluation relies on stratified five-fold cross-validation to avoid fragile point estimates from a tiny hold-out test set. Table 1 reports the dataset composition and the approximate per-fold split used in evaluation.

**Table 1: Dataset structure and approximate per-fold split used for stratified 5-fold cross-validation (counts per fold).**

| Class | Defect type | Total images | Train/Val/Test per fold (count) |
| :--- | :--- | :---: | :---: |
| 1 | Crack | 20 | 14/2/4 |
| 2 | Soiling | 10 | 7/1/2 |
| 3 | Delamination | 170 | 122/14/34 |
| **Total** | | **200** | **143/17/40** |

### 3.5. On-Board Deep Learning: YOLOv11 Segmentation

The UAV tier runs a lightweight deep learning model for defect detection and segmentation. A YOLO-family segmentation model was chosen as it offers a strong accuracy–speed trade-off and has mature deployment tooling. Specifically, YOLOv11-seg is used as the core network [34]. The model outputs bounding boxes and pixel-level masks for each detected defect. Masks are valuable in PV inspection because they enable estimation of defect area and shape, which improves prioritization and supports temporal tracking.

The detector is trained on the annotated dataset described in Table 1. To reduce overfitting under severe class imbalance, training leverages transfer learning and class-aware sampling: the network is initialized from a pretrained YOLO checkpoint and fine-tuned with strong geometric and photometric augmentations (random affine transforms, flips, blur, and contrast/brightness jitter), while minority-class instances (cracks and soiling) are oversampled during mini-batch construction. Public thermography datasets (STHS-277 and PVF-10) [31], [32] are used to broaden the thermal appearance distribution during fine-tuning.

During inference, the model produces a set of candidate instances $\{(b_k, M_k, s_k, c_k)\}$, where $b_k$ is a bounding box and $M_k$ is a mask. Non-maximum suppression (NMS) removes duplicate boxes, and an analogous mask-level suppression is applied to reduce overlapping instance masks. The post-processing stage is particularly important because UAV video streams contain multiple near-duplicate frames, and thermal palettes can produce color artifacts that trigger spurious detections.

### 3.6. Mask Similarity, Size-Based Routing, and Post-Processing

On-board inference yields instance masks, which must be filtered and merged to produce stable defect events. In this study, mask overlap is based on intersection-over-union (IoU):

$$
\mathrm{IoU}(A,B) = \frac{|A\cap B|}{|A\cup B|}.
$$

Mask-level suppression (Mask-NMS) removes duplicate instances when IoU exceeds a threshold, retaining the higher-confidence prediction. Mask-NMS is important in UAV inspection because adjacent frames often contain the same defect with slightly shifted masks.

Beyond duplicate suppression, the CPS performs size-based routing of defect instances between the two thermogram representations. Let $S_{\mathrm{def}}$ denote the area (in pixels) of a detected defect mask, and let $S_{\mathrm{cell}}$ denote the area of a single PV cell in the same image scale. The relative defect area is defined as follows:

$$
R_d = \frac{S_{\mathrm{def}}}{S_{\mathrm{cell}}}.
$$

The intuition is that defects smaller than one cell behave like localized hot spots and benefit from the higher contrast of M3, whereas larger anomalies are more reliably represented in the wider-range M2 thermogram.

Using $R_d$, two mutually exclusive defect sets are formed as follows:

$$
\mathcal{D}_{\mathrm{small}} = \{ d \in \mathcal{D}^{\mathrm{M3}} \mid R_d(d) < 1.0 \},
$$

$$
\mathcal{D}_{\mathrm{large}} = \{ d \in \mathcal{D}^{\mathrm{M2}} \mid R_d(d) \ge 1.0 \},
$$

and the final set of detections is:

$$
\mathcal{D}_{\mathrm{final}} = \mathcal{D}_{\mathrm{small}} \cup \mathcal{D}_{\mathrm{large}}.
$$

Because the routing criteria are complementary and Mask-NMS is applied within each branch, the resulting sets satisfy $\mathcal{D}^{\mathrm{M3}}\cap \mathcal{D}^{\mathrm{M2}}=\varnothing$ in practice, and duplicate alarms are reduced. Applying Mask-NMS and selective combination reduced false triggers by approximately 8% in field-like sequences.

Two palette-specific post-processing pipelines are distinguished in this research.

**M3 processing (three-color palette).** The three-color palette increases sensitivity to small anomalies but can introduce banding. The M3 branch therefore targets small defects by Eq. (7) and uses a conservative IoU threshold (0.4) in Mask-NMS. Algorithm 2 summarizes the steps.

**Algorithm 2: On-board processing of a three-color thermogram (M3)**
```text
Require: Thermogram I^th in three-color palette, YOLOv11-seg model, thresholds (τ_c, τ_IoU)
Ensure: Filtered defect set D_small
1: Convert thermogram to a multi-channel representation (palette split)
2: Run YOLOv11-seg to obtain candidate masks and confidences
3: Apply confidence filtering: keep masks with s_k >= τ_c (e.g., τ_c=0.25)
4: Compute R_d for each mask using Eq. (6); keep only instances with R_d < 1.0
5: Apply Mask-NMS using IoU threshold τ_IoU (e.g., 0.4)
6: Geo-tag detections using RTK navigation record n_t
7: return D_small
```

**M2 processing (two-color palette).** The two-color palette covers a wider temperature range and supports detection of larger anomalies. The M2 branch targets large defects by Eq. (8) and uses a slightly higher Mask-NMS IoU threshold (0.5). Algorithm 3 gives the steps.

**Algorithm 3: On-board processing of a two-color thermogram (M2)**
```text
Require: Thermogram I^th in two-color palette, YOLOv11-seg model
Ensure: Filtered defect set D_large
1: Normalize thermogram and run YOLOv11-seg
2: Apply confidence filtering and Mask-NMS (IoU threshold 0.5)
3: Compute R_d for each mask using Eq. (6); keep only instances with R_d >= 1.0
4: Geo-tag detections using RTK navigation record n_t
5: return D_large
```

### 3.7. Late-Fusion Ensemble Across Thermogram Representations

Palette-induced domain differences motivate model fusion. Rather than training a large multi-domain model, a lightweight late-fusion ensemble is implemented based on the size-routed branches defined in Eqs. (7)–(9). The detector is applied to both representations, producing $\mathcal{D}_t^{\mathrm{M2}}$ and $\mathcal{D}_t^{\mathrm{M3}}$, and the edge tier combines them by the set-union rule:

$$
\mathcal{D}_t^{\mathrm{ens}} \equiv \mathcal{D}_{\mathrm{final}} = \mathcal{D}_{\mathrm{small}} \cup \mathcal{D}_{\mathrm{large}}.
$$

This selective fusion keeps the branch that is best suited to the expected thermal contrast regime: M3 contributes the small-defect set, and M2 contributes the large-defect set. The edge tier further suppresses duplicates across frames using RTK geo-indexing (Section 3.8). The design is compatible with real-time operation because it does not require joint training or a heavier backbone; the additional cost is limited to running the detector on two representations.

### 3.8. Edge-Level Geo-Indexing and De-Duplication

A UAV mission produces repeated observations of the same modules across consecutive frames and overlapping flight lines. Without de-duplication, a single defect can be counted many times, inflating alarms and reducing operator trust. Therefore, RTK-assisted geo-indexing and de-duplication is performed at the edge tier.

Each detection in $\mathcal{D}_t$ is mapped to an approximate module location using the UAV RTK position, altitude, and camera geometry. The exact projection depends on camera calibration and gimbal orientation; in the current implementation, a simplified mapping sufficient for clustering is utilized at module granularity. Moreover, association is refined using the plant layout map.

To compute distances between geo-tagged detections, the Haversine distance between latitude/longitude coordinates is used in the following manner:

$$
d = 2R \arcsin\!\left(\sqrt{\sin^2\!\left(\frac{\Delta\varphi}{2}\right)+\cos\varphi_1\cos\varphi_2\sin^2\!\left(\frac{\Delta\lambda}{2}\right)}\right),
$$

where $R$ is the Earth radius, $\varphi$ is latitude, and $\lambda$ is longitude.

Detections are clustered using a distance threshold $\epsilon$ derived from the ground sampling distance and expected RTK error. Within each cluster, the edge server aggregates detections over time, retains representative evidence frames, and associates the cluster to a module ID or string based on the plant layout. The result is a set of module-level defect events rather than frame-level detections.

### 3.9. SCADA-Aware State Inference and Fire-Risk Logic

Defect detection becomes operationally useful when it maps to plant-level actions. The CPS includes a decision layer that combines UAV detections with electrical and supervisory signals to infer operating states and hazard conditions. The core idea is to express safety-relevant conditions in interpretable logic that can be audited by engineers and integrated into SCADA rule engines.

Binary variables are defined as $X_1$, $X_2$, and $X_3$. These variables indicate the presence of a defect in strings 1–3 of a module group (or a comparable partition of the electrical layout). Variables $X_{11}, X_{12}, X_{13}$ indicate whether bypass diodes in the corresponding strings operate correctly (1 indicates correct operation). The hazard output $Y$ is defined as:

$$
Y = X_1 \overline{X_{11}} + X_2 \overline{X_{12}} + X_3 \overline{X_{13}},
$$

which captures the engineering intuition that defects combined with failed bypass diodes increase overheating risk. The full truth report is summarized in Table 2.

To incorporate additional safety sensors (e.g., smoke or flame detectors), binary variables $X_{21}$ and $X_{22}$ are added; they represent sensor triggers. The extended hazard logic is:

$$
Y = X_1 \overline{X_{11}} + X_2 \overline{X_{12}} + X_3 \overline{X_{13}} + X_1 X_{21} + X_2 X_{22}.
$$

The edge server evaluates these expressions in real time and forwards alarms to the SCADA/HMI layer, where standard procedures (e.g., dispatch, shutdown recommendation) can be executed.

**Table 2: Truth table for module operating state and hazard output $Y$ (excerpted and translated). $X_1$–$X_3$ indicate defects in three strings; $X_{11}$–$X_{13}$ indicate bypass diode correctness; $Y=1$ denotes a hazardous state.**

| # | $X_1$ | $X_2$ | $X_3$ | $X_{11}$ | $X_{12}$ | $X_{13}$ | $Y$ | State description |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | normal operation |
| 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | normal operation |
| 3 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | normal operation |
| 4 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | normal operation |
| 5 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | normal operation |
| 6 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | normal operation |
| 7 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | normal operation |
| 8 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | normal operation |
| 9 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | hazard; bypass diode failure |
| 10 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | hazard; bypass diode failure |
| 11 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | hazard; bypass diode failure |
| 12 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | hazard; bypass diode failure |
| 13 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | hazard; bypass diode failure |
| 14 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | hazard; bypass diode failure |
| 15 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | hazard; bypass diode failure |
| 16 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | bypass diode blocks string |
| 17 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | bypass diode blocks string |
| 18 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | bypass diode blocks string |
| 19 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | bypass diodes block strings |
| 20 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | bypass diodes block strings |

### 3.10. Temperature Context via NOCT Approximation

To interpret thermal anomalies, the CPS can optionally estimate expected module temperature under nominal conditions using the Nominal Operating Cell Temperature (NOCT) approximation. Given ambient temperature $T_{\mathrm{amb}}$ and irradiance $G$ (W/m$^2$), the expected module temperature is approximated by:

$$
T_m = T_{\mathrm{amb}} + \frac{\mathrm{NOCT}-20}{800}\,G.
$$

This estimate is not a replacement for thermography; it provides context for whether observed temperatures are plausible under current irradiance and weather conditions, which can help to flag cases where reflections or transient shading dominate.

### 3.11. Experimental Setup

Experiments were executed in a closed-loop CPS workflow: (i) UAV sensing, (ii) on-board inference, (iii) edge aggregation, and (iv) SCADA-facing events, as outlined in Section 3.1. An RTK-enabled Matrice-class UAV was used as the sensing platform, carrying a gimbaled dual-sensor payload with a thermal infrared camera and a 4K RGB camera. The payload captures synchronized thermographic images and high-resolution RGB context frames, each geotagged by the UAV's centimeter-level RTK GNSS. Comparable UAV-based thermography configurations have been documented for PV plant inspection and quality control [35]. The gimbal's imaging system is driven by a dedicated low-power video processor (Ambarella H2 SoC) [36], which supports 4K 60 FPS video encoding. All image frames were acquired and pre-processed using OpenCV v4.9.0 [37] before inference.

On-board the UAV, a real-time object detection model runs continuously to identify PV panel anomalies from the thermal video feed. The Ultralytics v8.3.193 YOLO models [34] were deployed and implemented with the PyTorch v2.7.0 deep learning library [38]. Prior to deployment, the segmentation model was fine-tuned on the curated thermal dataset in Table 1 with stratified five-fold cross-validation; all reported detection metrics correspond to the mean $\pm$ standard deviation across folds unless stated otherwise. Training was conducted on a workstation equipped with a single NVIDIA RTX 3060 GPU (12 GB), using 300 epochs, input resolution 640, and batch size 16.

The inference is executed on an embedded AI computing platform integrated into the UAV payload. In our setup, the on-board inference engine leverages a Qualcomm QCS605 system-on-chip [39], which provides hardware-accelerated neural processing for edge vision tasks.

During flight, the UAV transmits compact detection messages, as defined in Equation (1), over a wireless link to a ground-based edge server. Meanwhile, full-resolution thermal and RGB frames are cached locally for post-flight audit and reporting. The edge server, powered by an NVIDIA Jetson Orin platform [40], aggregates incoming detections and performs spatial clustering to eliminate duplicates (Section 3.8). The clustering radius was set $\epsilon$ to accommodate the positioning uncertainty of the UAV's RTK navigation; specifically, $\epsilon$ was chosen in line with the ~2 cm accuracy attainable by low-cost dual-frequency RTK receivers like the u-blox ZED-F9P [41]. To maximize throughput and minimize latency, the YOLO inference model was optimized on the Jetson edge server using NVIDIA TensorRT v10.9.0 [42]. Finally, the consolidated and de-duplicated detection events are forwarded to the SCADA system in real time, enabling automated monitoring and operator alerts.

Defect detection performance is evaluated using standard detection metrics [43]. For a given class, precision and recall are utilized. Detection quality is summarized by average precision (AP) and mean average precision (mAP). In this study, mAP@0.5 is reported, i.e., AP computed at IoU threshold 0.5 and averaged across classes. For field validation against manual monitoring, the root mean squared error (RMSE) between automatic and manual defect counts is reported in defects per inspected PV string.

Operational performance is characterized by inference throughput (frames per second), end-to-end processing time, and sensitivity to flight and environmental parameters (altitude, speed, time of day, weather). In Section 4.5, the precision metric is used to report the fraction of correct defect detections among all raised detections under the chosen IoU/confidence thresholds.

## 4. Results

This section reports quantitative results for the proposed CPS-enabled defect detection pipeline. This study focuses on real data experiments and exclude synthetic experiments. Results are organized into (i) detection quality on two thermogram representations (M2 and M3), (ii) the effect of ensemble fusion, (iii) edge hardware considerations and YOLO model comparison, (iv) field validation against manual inspection, and (v) sensitivity to flight and weather parameters.

### 4.1. Detection Quality on Two Benchmark Thermogram Representations

Table 3 summarizes per-class results for YOLOv11-seg on M2 and M3, reported as mean $\pm$ standard deviation over stratified five-fold cross-validation. On the two-color palette (M2), the model achieves macro mAP@0.5 of 0.91 ± 0.03, precision of 0.89 ± 0.03, and recall of 0.87 ± 0.04. On the three-color palette (M3), macro mAP@0.5 is 0.90 ± 0.04 with precision 0.88 ± 0.04 and recall 0.86 ± 0.05. These averages are close, but class-wise behavior differs. Crack detection is stronger on M2 (0.93 ± 0.04 mAP@0.5) than on M3 (0.87 ± 0.07), while delamination detection is strongest on M3 (0.93 ± 0.02). This pattern supports the practical claim that thermogram palettes act as distinct domains and that a single representation may emphasize some defect signatures at the expense of others.

**Table 3: Comprehensive performance comparison of YOLOv11-seg on thermogram M2 (two-color palette) and M3 (three-color palette), reported as mean $\pm$ standard deviation over stratified 5-fold cross-validation.**

| Class | **M2 mAP@0.5** | **M2 Precision** | **M2 Recall** | **M3 mAP@0.5** | **M3 Precision** | **M3 Recall** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Crack | 0.93 ± 0.04 | 0.91 ± 0.05 | 0.89 ± 0.06 | 0.87 ± 0.07 | 0.86 ± 0.07 | 0.84 ± 0.08 |
| Soiling | 0.90 ± 0.06 | 0.88 ± 0.07 | 0.86 ± 0.08 | 0.90 ± 0.06 | 0.88 ± 0.07 | 0.86 ± 0.08 |
| Delamination | 0.89 ± 0.02 | 0.87 ± 0.02 | 0.85 ± 0.03 | 0.93 ± 0.02 | 0.90 ± 0.02 | 0.88 ± 0.03 |
| **Macro average** | **0.91 ± 0.03** | **0.89 ± 0.03** | **0.87 ± 0.04** | **0.90 ± 0.04** | **0.88 ± 0.04** | **0.86 ± 0.05** |

![Evaluation of detection quality (mean mAP@0.5 across five folds) for YOLOv11-seg across distinct thermogram representations. (a) The two-color palette (M2) excels in detecting cracks. (b) The three-color palette (M3) demonstrates superior performance for delamination.](figs/map_m2.pdf)
*Figure 3: Evaluation of detection quality (mean mAP@0.5 across five folds) for YOLOv11-seg across distinct thermogram representations.*

### 4.2. Effect of Late-Fusion Ensemble and Runtime Impact

Late-fusion ensemble improves robustness without requiring a heavier network. Table 4 reports mAP@0.5 before and after fusion (mean $\pm$ std over folds). The crack class benefits most, improving from 0.93 ± 0.04 to 0.96 ± 0.03 mAP@0.5. This matters operationally because crack-like patterns can be subtle and can be confounded by thermal noise or reflections. Delamination improves from 0.93 ± 0.02 to 0.95 ± 0.02, while soiling remains stable. Figure 4 visualizes these changes.

The ensemble pipeline also improves runtime. By merging detections early and avoiding repeated post-processing, the reported per-frame processing time was reduced from 4.235 s to 2.858 s. Although this latency depends on hardware and implementation, the reduction illustrates that careful pipeline design can improve both accuracy and operational responsiveness, which is critical for CPS operation.

**Table 4: Ensemble fusion effect on mAP@0.5 (before vs. after), reported as mean $\pm$ standard deviation over stratified 5-fold cross-validation.**

| Class | Before | After ensemble |
| :--- | :---: | :---: |
| Crack | 0.93 ± 0.04 | 0.96 ± 0.03 |
| Soiling | 0.90 ± 0.06 | 0.90 ± 0.06 |
| Delamination | 0.93 ± 0.02 | 0.95 ± 0.02 |

![mAP@0.5 improvements after late-fusion ensemble of M2 and M3 detections.](figs/ensemble_gain.pdf)
*Figure 4: mAP@0.5 improvements after late-fusion ensemble of M2 and M3 detections.*

### 4.3. Edge Hardware and YOLO Model Comparison

Deploying deep learning on UAVs imposes strict constraints on compute, weight, and power. Table 5 summarizes representative embedded platforms and highlights a practical trade-off: throughput controls the maximum feasible flight speed and frame rate. Importantly, detection quality (mAP/precision) is a property of the deployed model and numerical precision, not the hardware alone. Therefore, the mAP@0.5 values in Table 5 correspond to the platform-specific deployment configuration (e.g., FP16 vs. INT8 quantization and, when required, additional compression to meet memory/latency constraints).

**Table 5: Representative edge hardware comparison for on-board inference. Reported mAP@0.5 corresponds to the deployment configuration (engine and numerical precision), not the hardware alone.**

| Platform | Inference precision | Throughput (FPS) | mAP@0.5 | Precision |
| :--- | :---: | :---: | :---: | :---: |
| NVIDIA Jetson Orin Nano | FP16 (TensorRT) | 100 | 0.95 | 0.93 |
| Ambarella H2 | INT8 (compressed) | 60 | 0.70 | 0.75 |
| Qualcomm QCS605 | INT8 (SNPE) | 80 | 0.85 | 0.82 |

![Throughput vs. detection quality trade-off for representative embedded platforms.](figs/hardware_tradeoff.pdf)
*Figure 5: Throughput vs. detection quality trade-off for representative embedded platforms.*

To further justify the choice of YOLOv11, several YOLO generations were trained and evaluated under a consistent protocol. Table 6 reports precision, recall, and mAP@0.5, along with training time. YOLOv11 achieves the strongest overall performance (precision 0.96, recall 0.95, mAP@0.5 0.93), albeit with longer training time than some smaller models. The result suggests that, for on-board inference, YOLOv11 provides a favorable balance between accuracy and manageable deployment complexity.

**Table 6: Quantitative comparison of YOLO model generations (segmentation). Training time is measured for 300 epochs on a single NVIDIA RTX 3060 (12 GB) GPU with input resolution 640 and batch size 16.**

| Model | Precision | Recall | mAP@0.5 | Training time (min) |
| :--- | :---: | :---: | :---: | :---: |
| YOLOv5 | 0.92 | 0.90 | 0.86 | 55 |
| YOLOv8 | 0.91 | 0.88 | 0.83 | 45 |
| YOLOv9 | 0.94 | 0.80 | 0.90 | 59 |
| YOLOv10 | 0.94 | 0.86 | 0.90 | 132 |
| YOLOv11 | 0.96 | 0.95 | 0.93 | 106 |

### 4.4. Field Validation Against Manual Monitoring

A CPS intended for plant operation must agree with human inspection under field conditions. The CPS output is validated against manual monitoring on two PV installations: a rooftop plant and a ground-mounted plant. The evaluation compares automatic defect counts after edge-level de-duplication with manual counts obtained by expert review. The validation set comprised approximately 2,500 PV modules across two sites and contained 145 expert-confirmed defect instances; agreement is summarized by an RMSE of 0.71 defects per inspected PV string.

Beyond aggregate error, field testing highlights the role of CPS components that are not visible in offline metrics. On-board inference alone produces repeated detections for the same defect across adjacent frames; RTK-based clustering and de-duplication at the edge tier reduce this redundancy and produce module-level events suitable for reporting. Similarly, when abnormal heating is detected, the SCADA-aware logic can elevate the event priority, especially if bypass diode status indicates increased risk. These elements are essential for operational adoption because operators need stable counts, clear module identifiers, and actionable alerts rather than a stream of raw detections.

### 4.5. Sensitivity to Flight and Environmental Parameters

Thermographic inspection is physics-constrained. Therefore, flight altitude, speed, time of day, and weather are analyzed on how they affect detection quality. The results quantify the operating envelope of the proposed CPS.

Altitude creates a resolution–coverage trade-off. At 5 m altitude, detection precision reaches 98% with 96% recall, but coverage is limited and mission time increases. At 10 m, precision remains high (93%, recall 90%) while coverage increases by approximately 4–5 times, making it an attractive operational setting. At 15 m, precision drops to 84% (recall 79%), indicating that spatial resolution becomes insufficient for subtle defects.

Flight speed affects motion blur and temporal redundancy. At 3–7 m/s, precision remains above 91% and recall above 88%, suggesting that the pipeline tolerates moderate speeds when the camera shutter and stabilization are adequate. At 10 m/s, precision decreases to 85% and recall to 82%, which may be unacceptable for safety-critical monitoring.

Environmental conditions also matter. Midday inspection (12:00–14:00) performs best (96% precision, 94% recall), consistent with stronger irradiance and thermal contrast. Cloudy conditions improve performance (96% precision, 94% recall) compared with clear sky (92% precision, 89% recall), likely because diffuse illumination reduces specular reflections. These results align with prior observations that reflections can dominate thermograms under clear conditions and that careful flight planning is required [7], [11].

**Table 7: Sensitivity to altitude and speed (precision/recall).**

| Altitude (m) | Precision (%) | Recall (%) | | Speed (m/s) | Precision (%) | Recall (%) |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 5 | 98 | 96 | | 3 | 94 | 92 |
| 10 | 93 | 90 | | 5 | 93 | 90 |
| 15 | 84 | 79 | | 7 | 91 | 88 |
| | | | | 10 | 85 | 82 |

**Table 8: Sensitivity to time of day and weather (precision/recall).**

| Time Window | Precision (%) | Recall (%) | | Weather | Precision (%) | Recall (%) |
| :---: | :---: | :---: | :--- | :--- | :---: | :---: |
| 08:00–10:00 | 93 | 90 | | Clear | 92 | 89 |
| 12:00–14:00 | 96 | 94 | | Cloudy | 96 | 94 |
| 17:00–19:00 | 92 | 89 | | | | |

![Sensitivity of detection performance to operational and environmental variations. Results show that increasing (a) flight altitude and (b) speed degrades detection quality, necessitating strictly bounded flight profiles. Environmental factors also drive performance, with (c) the 12:00–14:00 window and (d) cloudy conditions providing optimal precision and recall.](figs/flight_height.pdf)
*Figure 6: Sensitivity of detection performance to operational and environmental variations.*

### 4.6. Quantitative Comparison with Representative State of the Art

The obtained results are compared with representative approaches reported in the literature. Direct comparison is imperfect because datasets, defect definitions, and evaluation protocols differ. Nevertheless, Table 9 provides a quantitative anchor and clarifies how the proposed CPS differs in scope.

The literature shows that high mAP can be achieved under favorable conditions, for example with ST-YOLO on IR images [18]. The proposed ensemble reaches comparable mAP@0.5 for the crack class (0.96) while adding system-level components required for plant operation: geo-indexing, de-duplication, and SCADA-aware decision logic. Multi-stage approaches such as Di Tommaso et al. [14] report substantially lower anomaly detection AP@0.5 on hotspots (66.9%), which illustrates that anomaly detection can be harder than module detection and that dataset and defect type strongly influence metrics. Dotenco et al. [25] reported strong defect classification performance using classical vision and statistical tests, which remains relevant as a baseline and demonstrates that robust pre-processing is crucial in aerial IR imagery.

**Table 9: Representative quantitative comparison with published approaches (as reported by the respective authors).**

| Work | Modality | Model / approach | Task | Reported metric |
| :--- | :--- | :--- | :--- | :--- |
| Dotenco et al. [25] | IR | Statistical tests + vision | Defect classification | F1 = 93.88% |
| Di Tommaso et al. [14] | IR | Multi-stage YOLOv3 | Hotspot detection | AP@0.5 = 66.9% |
| Xie et al. [18] | IR | ST-YOLO | PV fault detection | mAP@0.5 = 96.6% |
| **This work** | **IR + RGB** | **YOLOv11-seg + ensemble + CPS** | **Defect segmentation + decision** | **mAP@0.5 up to 96% (cracks)** |

![Representative quantitative comparison (reported metrics). Values are not strictly comparable across datasets; the figure contextualizes the magnitude of results.](figs/sota_comparison.pdf)
*Figure 7: Representative quantitative comparison (reported metrics).*

## 5. Discussion

The presented results support two intertwined claims. First, a modern segmentation-capable detector can achieve strong defect localization on thermal UAV imagery even with a modest training dataset, provided that post-processing and operational constraints are respected. Second, and more importantly, the operational usefulness of UAV inspection depends on a cyber-physical architecture that connects detection outputs to geo-referenced, de-duplicated events and to plant supervision.

From a computer vision standpoint, the palette-specific results illustrate a subtle but important phenomenon: the thermogram representation is part of the model's input domain. A two-color palette (M2) emphasizes certain gradients and yields stronger crack detection, while a three-color palette (M3) strengthens delamination detection. This is consistent with broader observations in PV thermography that acquisition and rendering choices influence defect contrast and therefore detectability [6], [4]. The late-fusion ensemble improves robustness by exploiting this complementarity. The improvement for cracks (from 0.93 to 0.96 mAP@0.5) is particularly relevant because cracks are often early-stage defects whose thermal signatures can be weak. The runtime reduction (4.235 s to 2.858 s) further highlights that engineering the pipeline can yield tangible benefits.

From a systems standpoint, the edge tier is not a luxury. Without RTK-based geo-indexing and de-duplication, raw UAV detections would produce inflated counts and unstable alarms. The architecture therefore treats geo-referencing as a first-class signal, consistent with field studies that emphasize the practical difficulty of mapping anomalies back to modules and strings [7], [11]. The edge tier also provides a natural integration point for SCADA signals such as bypass diode status. The Boolean hazard logic (Eqs. (12)–(13)) is intentionally interpretable: plant engineers can audit it, adjust it, and embed it in existing rule engines. This contrasts with end-to-end "black box" alarm models, which may be difficult to certify for safety monitoring.

The sensitivity analysis offers another argument for the CPS framing. The sharp performance drop at 15 m altitude and at 10 m/s speed shows that the detector's performance is constrained by physics and optics, not only by model capacity. Similarly, better performance under cloudy conditions suggests that reflections can be a major source of false alarms. These observations motivate the inclusion of reflection-aware planning (Section 3.3) and the use of contextual temperature models (Section 3.10) as part of the overall system. In other words, the "best model" in isolation is insufficient; the system must actively shape its own data through flight planning and quality control.

A key limitation of the present evaluation is the small and imbalanced labeled dataset for the minority classes. Even with stratified five-fold cross-validation, the crack and soiling classes contain only 20 and 10 labeled images, respectively, which leads to higher variance (Tables 3–4). The reported means should therefore be interpreted as indicative of feasibility rather than as definitive benchmarks. Expanding labeled datasets and adopting standardized public UAV thermography benchmarks with consistent metadata would substantially improve statistical confidence and reproducibility.

The proposed approach also has disadvantages and trade-offs. The architecture introduces deployment complexity: an edge server must be maintained on-site, and reliable communication between UAV and edge must be ensured. The approach depends on RTK positioning and on consistent camera calibration; failure in these subsystems can degrade geo-indexing and reduce trust in module-level mapping. Although transfer learning and augmentation mitigate overfitting, generalization to new defect types, new thermal cameras, or different palettes remains a challenge. Future work should investigate palette-invariant representations or temperature-calibrated models that operate directly on radiometric data rather than rendered palettes.

Several limitations point to open research directions. First, thermography remains sensitive to environmental conditions; a key challenge is to incorporate irradiance, wind, and ambient temperature into learning models in a principled way, potentially through physics-informed features or uncertainty-aware inference. Second, the current SCADA logic relies on explicit Boolean rules; hybrid approaches that combine learned risk scoring with interpretable rules could improve flexibility while preserving auditability. Third, the geo-indexing step currently uses simplified projection; improving geometric modeling (e.g., through camera calibration and plant 3D models) could reduce association errors in dense plants. Finally, there is a broader question of benchmarking: the PV community would benefit from standardized, publicly available UAV thermography datasets with consistent labels and metadata (flight altitude, irradiance, camera parameters) to enable reproducible comparison across systems.

## 6. Conclusion

This study establishes a robust cyber-physical architecture for photovoltaic plant monitoring that successfully bridges the gap between raw computer vision detections and actionable operational maintenance. By integrating real-time UAV-based YOLOv11 segmentation with an intelligent edge-computing tier for RTK-assisted geo-indexing and duplicate suppression, the system effectively transforms fragmented inference outputs into traceable, module-level defect events. The experimental results confirm the superior efficacy of the proposed late-fusion ensemble, which notably improved crack detection to 0.96 mAP@0.5 and delamination detection to 0.95 mAP@0.5, while simultaneously reducing per-frame processing latency to 2.858 s. Field validation further demonstrated alignment with manual inspection records, achieving an RMSE of 0.71 defects per inspected PV string and identifying a 10 m flight altitude as the most balanced operating point with 93% precision and 90% recall. Uniquely, the integration of interpretable SCADA-aware logic allows for immediate fire-risk assessment based on bypass diode status. Despite these successes, the noteworthy limitations of this study include sensitivity to specular reflections under clear sky conditions, performance degradation at flight speeds exceeding 7 m/s, and statistically fragile evaluation for minority defect classes due to limited labeled data. Reliance on precise RTK calibration for module association also presents a deployment challenge in dense array configurations. Overall, the proposed architecture offers a scalable, safety-centric solution for automating the lifecycle management of large-scale solar assets.

Future research will prioritize palette-invariant radiometric models to mitigate environmental noise, larger benchmark-driven evaluations to reduce uncertainty, and predictive maintenance algorithms that leverage historical SCADA trends.

## References

[1] sun2022photovoltaic (Citation details not provided in source)
[2] sinha2024review (Citation details not provided in source)
[3] lynnyk2020ddos (Citation details not provided in source)
[4] aghaei2025autonomous (Citation details not provided in source)
[5] khatri2025comprehensive (Citation details not provided in source)
[6] gallardo2020infrared (Citation details not provided in source)
[7] prasshanth2025fault (Citation details not provided in source)
[8] moctezuma2025deep (Citation details not provided in source)
[9] svystun2025dytam (Citation details not provided in source)
[10] rouibah2025smart (Citation details not provided in source)
[11] rodriguez2024real (Citation details not provided in source)
[12] noura2024explainable (Citation details not provided in source)
[13] bodyanskiy2022deep (Citation details not provided in source)
[14] ditommaso2022multistage (Citation details not provided in source)
[15] melnychenko2024intelligent (Citation details not provided in source)
[16] aljafari2024supervised (Citation details not provided in source)
[17] setiawan2025exploring (Citation details not provided in source)
[18] xie2024st (Citation details not provided in source)
[19] carni2017distributed (Citation details not provided in source)
[20] boucif2025artificial (Citation details not provided in source)
[21] lysyi2025enhanced (Citation details not provided in source)
[22] tang2022deep (Citation details not provided in source)
[23] ferlito2024study (Citation details not provided in source)
[24] masita2025deep (Citation details not provided in source)
[25] dotenco2016automatic (Citation details not provided in source)
[26] ebied2025advanced (Citation details not provided in source)
[27] barraz2025fast (Citation details not provided in source)
[28] abdelsattar2025resnet (Citation details not provided in source)
[29] svystun2024thermal (Citation details not provided in source)
[30] zheng2025wavelet (Citation details not provided in source)
[31] Alfaro2019Dataset (Citation details not provided in source)
[32] Wang2024PVF10 (Citation details not provided in source)
[33] cvat2024cvat (Citation details not provided in source)
[34] jocher2025ultralytics (Citation details not provided in source)
[35] ec2018standards (Citation details not provided in source)
[36] ambarella2016low (Citation details not provided in source)
[37] bradski2000opencv (Citation details not provided in source)
[38] paszke2019pytorch (Citation details not provided in source)
[39] qualcomm2020qualcomm (Citation details not provided in source)
[40] nvidia2024jetson (Citation details not provided in source)
[41] robustelli2023low (Citation details not provided in source)
[42] nvidia2024tensorrt (Citation details not provided in source)
[43] rainio2024evaluation (Citation details not provided in source)
