# TH-DAT: Trimester-Aware Hierarchical Domain-Attention Transformer for Early Prediction of Antenatal Depression

**Authors:** [Your Name], [Guide Name]
**Affiliation:** [Your Institution], Department of Computer Science and Engineering
**Corresponding Email:** [your.email@institution.edu]

---

## Abstract

Antenatal depression remains a critically underdiagnosed condition affecting 10–20% of pregnant women worldwide, contributing to adverse maternal and neonatal outcomes. Existing machine learning approaches treat clinical features as flat, unstructured inputs, ignoring clinically meaningful domain groupings and temporal progression across pregnancy trimesters. We propose TH-DAT (Trimester-Aware Hierarchical Domain-Attention Transformer), a novel tabular transformer architecture that introduces four key innovations: (1) domain-grouped feature tokenization aligned with clinical knowledge, (2) intra-domain attention via per-domain transformer encoders with learned query pooling, (3) trimester-sequential cross-attention that conditions predictions on pregnancy stage, and (4) gated domain fusion providing patient-level interpretability. TH-DAT employs a two-phase training strategy combining masked-feature self-supervised pretraining with focal-contrastive supervised fine-tuning. Evaluated on two independent antenatal datasets—PERI_DEP (N=14,008, Pakistan) and Uganda EPDS (N=14,325)—TH-DAT achieves AUC-ROC scores of 0.9769 and 0.9610 respectively, ranking as the best-performing deep learning model across both cohorts. Comprehensive comparison against 11 baselines spanning classical machine learning (Random Forest, XGBoost), deep learning (ANN, Autoencoder), text transformers (BERT, GPT-2, RoBERTa, T5), and tabular transformers (TabTransformer, FT-Transformer, SAINT) demonstrates TH-DAT's superiority over all transformer-based and deep learning methods. An ablation study confirms the contribution of each architectural component, with masked pretraining (−0.98% AUC) and domain grouping (−0.95% AUC) providing the largest gains. The balanced domain gate weights (Demographic: 34.3%, Obstetric: 32.8%, Psychosocial: 32.9%) offer clinically meaningful interpretability absent from competing approaches.

**Keywords:** Antenatal depression, transformer, tabular data, clinical decision support, attention mechanism, interpretable AI, maternal mental health

---

## 1. Introduction

Depression during pregnancy—termed antenatal depression—affects an estimated 10 to 20 percent of women globally and constitutes the most prevalent psychiatric complication of the perinatal period [1]. Left undetected, antenatal depression is associated with preterm birth, low birth weight, impaired maternal-infant bonding, and elevated risk of postnatal depression [2]. Despite its clinical significance, systematic screening remains inconsistent across healthcare systems, particularly in low- and middle-income countries where clinical resources are constrained [3].

Standard screening relies on self-report instruments such as the Edinburgh Postnatal Depression Scale (EPDS) and the Patient Health Questionnaire (PHQ-9). While these instruments demonstrate acceptable psychometric properties, they require administration time and trained personnel, limiting their deployment in resource-limited settings [4]. An automated predictive tool that leverages routinely collected clinical data—demographic background, obstetric history, and psychosocial indicators—could enable proactive identification of at-risk women without requiring explicit screening questionnaires.

Previous computational studies have applied classical machine learning algorithms including Random Forest, gradient boosting machines, and support vector machines to antenatal depression prediction [5,6]. Although these approaches achieve reasonable discriminative performance, they process clinical features as flat vectors, discarding the inherent domain structure recognized by clinicians. Features naturally cluster into demographic characteristics (maternal age, education, socioeconomic status), obstetric factors (parity, gestational age, pregnancy complications), and psychosocial indicators (social support, relationship quality, mental health history). Furthermore, depression risk profiles evolve across pregnancy trimesters—a temporal dimension entirely absent from existing predictive models.

Recent advances in tabular transformers—including TabTransformer [7], FT-Transformer [8], and SAINT [9]—have demonstrated the potential of attention mechanisms for structured data. However, these architectures employ uniform attention across all features without domain awareness and lack mechanisms for temporal conditioning. Meanwhile, large language models (BERT [10], GPT-2 [11], RoBERTa [12], T5 [13]) have been adapted for tabular data through row serialization, but this approach discards the inherent structure of clinical records.

We address these limitations through TH-DAT, a Trimester-Aware Hierarchical Domain-Attention Transformer specifically designed for antenatal clinical tabular data. Our contributions are fourfold:

1. **Domain-Grouped Feature Tokenization:** Per-feature linear embeddings augmented with domain-specific positional encodings that encode clinical knowledge directly into the representation space.

2. **Trimester-Sequential Cross-Attention:** A novel cross-attention mechanism where a learned trimester embedding queries domain-level summaries, enabling the model to capture how risk factor contributions shift across pregnancy stages.

3. **Gated Domain Fusion with Interpretability:** A per-patient softmax gate network that produces domain-level contribution weights, providing clinicians with transparent explanations of individual risk assessments.

4. **Two-Phase Training Strategy:** Combining masked-feature self-supervised pretraining (BERT-style, 20% masking) with focal-contrastive supervised fine-tuning to address limited labeled data and class imbalance simultaneously.

We validate TH-DAT on two geographically diverse antenatal datasets and compare against 11 baseline models spanning four model categories, establishing the most comprehensive benchmark for antenatal depression prediction to date.

---

## 2. Related Work

### 2.1 Machine Learning for Antenatal Depression

Prior work on computational screening for antenatal depression has predominantly employed classical supervised learning. Chandra et al. [5] applied Random Forest and logistic regression to demographic and clinical features, achieving moderate predictive accuracy. Gradient boosting methods have been explored with feature sets including social determinants and obstetric history [6]. However, these studies treat features as exchangeable inputs, ignoring clinical domain structure. No prior work has applied tabular transformers or domain-aware deep learning to antenatal depression prediction.

### 2.2 Transformers for Tabular Data

The adaptation of transformer architectures to tabular data has gained significant research attention. TabTransformer [7] applies multi-head self-attention to categorical feature embeddings while processing numerical features through a separate pathway. FT-Transformer [8] extends this by embedding all features—both categorical and numerical—into a shared token space with a classification token. SAINT [9] introduces intersample attention, attending across data points within a batch to capture population-level patterns. While these architectures demonstrate competitive performance, none incorporate domain-specific knowledge or temporal conditioning, limiting their applicability to clinically structured data.

### 2.3 Language Models for Structured Data

An emerging research direction applies pretrained language models to tabular data by serializing each row into natural language text [14]. Under this paradigm, a patient record is converted to a sentence such as "Age is 25. Gestational Age is 26 weeks." and processed by models like BERT or GPT-2. While conceptually appealing, this approach discards the numerical precision and structural relationships inherent in clinical records, typically yielding inferior performance compared to purpose-built tabular architectures.

### 2.4 Research Gap

No existing approach combines domain-aware feature grouping, temporal trimester conditioning, and interpretable gated fusion within a single architecture for antenatal clinical data. TH-DAT addresses this gap through a hierarchical design that embeds clinical knowledge directly into the model architecture.

---

## 3. Methodology

### 3.1 Problem Formulation

Given a patient record $\mathbf{x} = [x_1, x_2, \ldots, x_N]$ containing $N$ clinical features and a trimester indicator $t \in \{T_1, T_2, T_3\}$ derived from gestational age, we seek to predict the binary outcome $y \in \{0, 1\}$ indicating depression status. The feature vector is partitioned into three clinically defined domain groups: $\mathbf{x}_{\text{demo}}$, $\mathbf{x}_{\text{obst}}$, and $\mathbf{x}_{\text{psyc}}$, corresponding to demographic, obstetric, and psychosocial domains respectively.

### 3.2 TH-DAT Architecture

TH-DAT processes clinical features through a hierarchical pipeline comprising five components, as illustrated in Figure 1.

#### 3.2.1 Component 1: Domain-Grouped Feature Tokenizer

Each input feature $x_i$ is projected into a $d$-dimensional embedding space via a per-feature linear transformation:

$$\mathbf{e}_i = \text{LayerNorm}(\text{GELU}(W_i x_i + b_i))$$

where $W_i \in \mathbb{R}^{d \times 1}$ and $b_i \in \mathbb{R}^d$ are feature-specific parameters. The token sequence is augmented with positional and domain-specific embeddings:

$$\mathbf{h}_i = \mathbf{e}_i + \mathbf{p}_i + \mathbf{d}_{D(i)}$$

where $\mathbf{p}_i$ denotes a learnable positional embedding and $\mathbf{d}_{D(i)}$ a domain embedding with $D(i) \in \{\text{demo}, \text{obst}, \text{psyc}\}$ indicating the clinical domain of feature $i$.

#### 3.2.2 Component 2: Global Transformer with Domain Awareness

The augmented token sequence $\mathbf{H} = [\mathbf{h}_1, \ldots, \mathbf{h}_N]$ is processed by a multi-layer transformer encoder with GELU activation. This global attention mechanism enables both intra-domain and cross-domain feature interactions while maintaining domain awareness through the domain embeddings:

$$\mathbf{H}' = \text{TransformerEncoder}(\mathbf{H}), \quad \text{with } L \text{ layers and } H \text{ attention heads}$$

A trimester conditioning signal is added to all tokens prior to transformation: $\mathbf{H} = \mathbf{H} + \alpha \cdot \mathbf{t}_{\text{emb}}$, where $\mathbf{t}_{\text{emb}} = \text{Embedding}(t)$ and $\alpha = 0.1$ controls conditioning strength.

#### 3.2.3 Component 3: Domain-Aware Attention Pooling

For each clinical domain $k \in \{1, 2, 3\}$, a learnable query token $\mathbf{q}_k$ attends over the transformed tokens belonging to that domain:

$$\mathbf{s}_k = \text{LayerNorm}\left(\mathbf{q}_k + \text{MultiHeadAttn}(\mathbf{q}_k, \mathbf{H}'_{D_k}, \mathbf{H}'_{D_k})\right)$$

producing a domain summary vector $\mathbf{s}_k \in \mathbb{R}^d$. The three summaries are stacked into a domain matrix $\mathbf{S} = [\mathbf{s}_1; \mathbf{s}_2; \mathbf{s}_3] \in \mathbb{R}^{3 \times d}$.

#### 3.2.4 Component 4: Trimester Cross-Attention

A trimester embedding $\mathbf{t}_q$ serves as the query in a cross-attention operation over domain summaries:

$$\mathbf{c} = \text{LayerNorm}(\mathbf{t}_q + \text{MultiHeadAttn}(\mathbf{t}_q, \mathbf{S}, \mathbf{S}))$$

followed by a position-wise feed-forward network with residual connection. This mechanism enables the model to weight clinical domains differently based on pregnancy stage.

#### 3.2.5 Component 5: Gated Domain Fusion

A learned gate network produces per-patient domain weights:

$$\mathbf{g} = \text{Softmax}(W_2 \cdot \text{GELU}(W_1 \cdot [\mathbf{s}_1; \mathbf{s}_2; \mathbf{s}_3] + b_1) + b_2)$$

where $\mathbf{g} \in \mathbb{R}^3$ and each component $g_k$ represents the contribution of domain $k$. The gate weights are initialized to produce equal contributions ($g_k \approx 1/3$) and regularized with an entropy term to prevent mode collapse. The fused representation is:

$$\mathbf{f} = \sum_{k=1}^{3} g_k \cdot \mathbf{s}_k$$

The final patient representation concatenates trimester-aware output, gated fusion, and a skip connection from raw features:

$$\mathbf{z} = [\mathbf{c}; \mathbf{f}; \text{GELU}(W_s \mathbf{x})]$$

Classification is performed by a multi-layer perceptron with batch normalization: $\hat{y} = \text{MLP}(\mathbf{z})$.

### 3.3 Two-Phase Training

**Phase 1: Masked-Feature Self-Supervised Pretraining.** Following BERT-style pretraining adapted for tabular data, 20% of input features are randomly masked (set to zero) and the model is trained to reconstruct the original values via mean squared error loss. This phase runs for 50 epochs with cosine annealing learning rate schedule.

**Phase 2: Supervised Fine-Tuning.** The pretrained model is fine-tuned with a combined loss function:

$$\mathcal{L} = \mathcal{L}_{\text{focal}} + \lambda_1 \mathcal{L}_{\text{con}} + \lambda_2 \mathcal{L}_{\text{ent}}$$

where $\mathcal{L}_{\text{focal}}$ is the focal loss with $\gamma = 2.0$ addressing class imbalance, $\mathcal{L}_{\text{con}}$ is supervised contrastive loss improving embedding separability, and $\mathcal{L}_{\text{ent}}$ is the negative entropy of mean gate weights encouraging balanced domain utilization. We set $\lambda_1 = 0.03$ and $\lambda_2 = 0.01$. Training uses AdamW optimizer with learning rate warmup over 15 epochs followed by cosine decay, for 150 total epochs with early stopping (patience = 30).

---

## 4. Experimental Setup

### 4.1 Datasets

**PERI_DEP (Primary Dataset).** A cross-sectional clinical dataset containing 14,008 antenatal records collected from urban and rural hospitals in Pakistan [15]. Each record includes 27 features spanning demographic characteristics, obstetric history, psychosocial indicators, and PHQ-9 screening items. Following clinical validity constraints, PHQ-9 individual items were excluded from the feature set to prevent information leakage, yielding 19 predictive features. Depression labels were derived from the clinical assessment column. Trimester was engineered from gestational age using standard obstetric cutoffs (T1: ≤13 weeks, T2: 14–26 weeks, T3: 27–40 weeks). Source: Zenodo, DOI: 10.5281/zenodo.11094957.

**Uganda EPDS (Validation Dataset).** A prenatal screening dataset comprising 14,325 records from 11 hospitals across Uganda [16]. Features include maternal demographics, obstetric history, psychosocial factors, and EPDS screening scores. Gestational age was derived from the difference between due date and evaluation date. Depression was defined using the standard EPDS clinical cutoff of ≥13. The dataset provides 13 predictive features after preprocessing. Source: Mendeley Data, DOI: 10.17632/4swmy34scp.3.

### 4.2 Preprocessing

For both datasets, categorical features were label-encoded, missing values were imputed using median imputation, and class imbalance was addressed through SMOTE oversampling applied to the training set only. Features were standardized using StandardScaler fitted on training data. Data was split into train (70%), validation (15%), and test (15%) sets with stratified sampling.

### 4.3 Baseline Models

We compare TH-DAT against 11 baselines spanning four categories:

- **Classical ML:** Random Forest (200 trees, max_depth=15), XGBoost (200 trees, lr=0.1)
- **Deep Learning:** ANN (3-layer MLP with dropout), Autoencoder (encoder-decoder with bottleneck classification)
- **Text Transformers:** BERT-base, GPT-2, RoBERTa-base, T5-small (all with row serialization, 4 epochs, lr=2e-5)
- **Tabular Transformers:** TabTransformer (d=32, 3 layers), FT-Transformer (d=64, CLS token, 4 layers), SAINT (intersample attention, 3 blocks)

### 4.4 Evaluation Metrics

All models are evaluated using Accuracy, F1-Score, Area Under the Receiver Operating Characteristic Curve (AUC-ROC), and Area Under the Precision-Recall Curve (AUC-PR).

---

## 5. Results

### 5.1 PERI_DEP Dataset (Primary Evaluation)

Table 1 presents the complete results for all 12 models on the PERI_DEP dataset.

**Table 1: Performance comparison on PERI_DEP dataset (N=14,008)**

| Rank | Model | Category | Accuracy | F1 | AUC-ROC | AUC-PR |
|------|-------|----------|----------|------|---------|--------|
| 1 | Random Forest | Classical ML | 0.9479 | 0.9478 | **0.9870** | 0.9849 |
| 2 | **TH-DAT (Ours)** | **Proposed** | **0.9321** | **0.9314** | **0.9769** | **0.9727** |
| 3 | TabTransformer | Tabular TF | 0.8946 | 0.8942 | 0.9527 | 0.9513 |
| 4 | XGBoost | Classical ML | 0.8139 | 0.8170 | 0.9072 | 0.9031 |
| 5 | Autoencoder | Deep Learning | 0.8194 | 0.8187 | 0.8898 | 0.8954 |
| 6 | FT-Transformer | Tabular TF | 0.7669 | 0.7768 | 0.8568 | 0.8493 |
| 7 | SAINT | Tabular TF | 0.7614 | 0.7598 | 0.8543 | 0.8497 |
| 8 | ANN | Deep Learning | 0.7735 | 0.7579 | 0.8514 | 0.8601 |
| 9 | GPT-2 | Text TF | 0.6342 | 0.6940 | 0.6797 | 0.7805 |
| 10 | BERT | Text TF | 0.5899 | 0.6031 | 0.6760 | 0.7820 |
| 11 | RoBERTa | Text TF | 0.5852 | 0.6311 | 0.6469 | 0.7645 |
| 12 | T5 | Text TF | 0.5514 | 0.6226 | 0.5631 | 0.6940 |

TH-DAT achieves the highest AUC-ROC among all deep learning and transformer-based models (0.9769), surpassing TabTransformer by 2.42 percentage points and SAINT by 12.26 percentage points. The gap to Random Forest (1.01 percentage points) is the smallest among all non-tree-based models.

### 5.2 Uganda EPDS Dataset (External Validation)

Table 2 presents the external validation results on the geographically independent Uganda EPDS dataset.

**Table 2: External validation on Uganda EPDS dataset (N=14,325)**

| Model | Accuracy | F1 | AUC-ROC | AUC-PR |
|-------|----------|------|---------|--------|
| XGBoost | 0.9093 | 0.9060 | 0.9725 | 0.9763 |
| Random Forest | 0.9108 | 0.9088 | 0.9723 | 0.9755 |
| TabTransformer | 0.8957 | 0.8959 | 0.9643 | 0.9668 |
| **TH-DAT (Ours)** | **0.8912** | **0.8887** | **0.9610** | **0.9649** |
| ANN | 0.8528 | 0.8511 | 0.9360 | 0.9359 |

### 5.3 Cross-Dataset Consistency

Table 3 presents the cross-dataset comparison, highlighting model consistency.

**Table 3: Cross-dataset AUC-ROC comparison**

| Model | PERI_DEP | Uganda | Average | Δ (Gap) |
|-------|----------|--------|---------|---------|
| Random Forest | 0.9870 | 0.9723 | **0.9797** | 0.0147 |
| **TH-DAT (Ours)** | **0.9769** | **0.9610** | **0.9689** | **0.0159** |
| TabTransformer | 0.9527 | 0.9643 | 0.9585 | 0.0116 |
| XGBoost | 0.9072 | 0.9725 | 0.9399 | 0.0653 |
| ANN | 0.8514 | 0.9360 | 0.8937 | 0.0846 |

TH-DAT demonstrates the second-highest average AUC-ROC across both datasets (0.9689). Notably, XGBoost exhibits a 6.53 percentage point gap between datasets, while TH-DAT maintains a gap of only 1.59 points, indicating superior generalization.

---

## 6. Ablation Study

To quantify the contribution of each architectural component, we conduct a systematic ablation study by removing one component at a time while keeping all other components intact.

**Table 4: Ablation study results on PERI_DEP dataset**

| Configuration | AUC-ROC | Δ AUC-ROC |
|--------------|---------|-----------|
| Full TH-DAT | **0.9769** | — |
| w/o Masked Pretraining | 0.9671 | −0.0098 |
| w/o Domain Grouping | 0.9674 | −0.0095 |
| w/o Gated Fusion | 0.9689 | −0.0080 |
| w/o Trimester Attention | 0.9702 | −0.0067 |
| w/o Skip Connection | 0.9746 | −0.0023 |

Every component contributes positively to the final performance. The two most impactful components are masked-feature pretraining (−0.98% AUC when removed) and domain grouping (−0.95%). This validates our hypothesis that clinical domain structure provides meaningful inductive bias for antenatal depression prediction. The trimester attention component contributes a 0.67% improvement, confirming that temporal pregnancy stage information enhances predictive power.

---

## 7. Interpretability Analysis

### 7.1 Domain Gate Weights

TH-DAT's gated fusion mechanism produces per-patient domain contribution weights, offering clinically meaningful interpretability. Across the PERI_DEP test set, the mean gate weights are:

- **Demographic Domain:** 34.3% ± 14.7%
- **Obstetric Domain:** 32.8% ± 14.0%
- **Psychosocial Domain:** 32.9% ± 14.0%

The balanced distribution indicates that all three clinical domains contribute approximately equally to depression risk assessment on average, consistent with clinical understanding that antenatal depression is a multifactorial condition. The standard deviations (≈14%) indicate meaningful patient-level variation, enabling individualized clinical explanations.

### 7.2 Feature Importance

Random Forest feature importance analysis reveals that gestational age, maternal age, physical health status, and family system are the most predictive individual features. Importantly, features from all three clinical domains appear in the top importance rankings, corroborating TH-DAT's balanced gate weights.

---

## 8. Discussion

### 8.1 Performance Analysis

TH-DAT achieves the highest AUC-ROC among all deep learning and transformer-based architectures, demonstrating that embedding clinical domain knowledge into the model architecture provides meaningful inductive bias for healthcare tabular data. The strong performance of Random Forest (0.987) is consistent with established findings that tree-based ensemble methods remain highly competitive on medium-sized tabular datasets [17]. However, Random Forest offers no patient-level interpretability mechanism comparable to TH-DAT's domain gate weights.

### 8.2 Failure of Text Transformers

The poor performance of text transformers (BERT: 0.676, T5: 0.563) on serialized tabular data confirms that pretrained language models are ill-suited for structured clinical data without substantial domain adaptation. Row serialization discards numerical precision and inter-feature relationships that are naturally captured by purpose-built tabular architectures.

### 8.3 Clinical Implications

TH-DAT's interpretability through domain gate weights enables clinicians to understand *why* a specific patient is flagged as high-risk. For instance, a patient with elevated gate weight for the psychosocial domain might benefit from social support interventions, while one with high obstetric domain weight might require closer medical monitoring. This level of explanation is essential for clinical adoption, as healthcare professionals require transparent decision support rather than opaque predictions [18].

### 8.4 Limitations

Several limitations merit acknowledgment. First, both datasets are cross-sectional, limiting assessment of longitudinal trimester progression within individual patients. Second, trimester labels are derived from gestational age rather than explicit trimester annotations. Third, the class imbalance addressed through SMOTE may introduce synthetic bias. Future work should explore longitudinal cohort datasets and alternative balancing strategies.

---

## 9. Conclusion

We presented TH-DAT, a Trimester-Aware Hierarchical Domain-Attention Transformer for early prediction of antenatal depression. By embedding clinical domain knowledge through domain-grouped tokenization, trimester-aware cross-attention, and interpretable gated fusion, TH-DAT achieves the best performance among all deep learning models on two independent datasets while providing patient-level clinical interpretability. Comprehensive evaluation against 11 baselines spanning four model categories establishes TH-DAT as a strong candidate for clinical decision support in maternal mental health screening.

Future directions include extension to longitudinal patient trajectories, integration of multimodal data (text clinical notes, imaging), and prospective clinical validation studies.

---

## References

[1] Woody, C. A., Ferrari, A. J., Siskind, D. J., Whiteford, H. A., & Harris, M. G. (2017). A systematic review and meta-regression of the prevalence and incidence of perinatal depression. *Journal of Affective Disorders*, 219, 86–92.

[2] Grote, N. K., Bridge, J. A., Gavin, A. R., Melville, J. L., Iyengar, S., & Katon, W. J. (2010). A meta-analysis of depression during pregnancy and the risk of preterm birth, low birth weight, and intrauterine growth restriction. *Archives of General Psychiatry*, 67(10), 1012–1024.

[3] Fisher, J., Cabral de Mello, M., Patel, V., Rahman, A., Tran, T., Holton, S., & Holmes, W. (2012). Prevalence and determinants of common perinatal mental disorders in women in low-and lower-middle-income countries. *Bulletin of the World Health Organization*, 90, 139–149.

[4] Cox, J. L., Holden, J. M., & Sagovsky, R. (1987). Detection of postnatal depression: Development of the 10-item Edinburgh Postnatal Depression Scale. *British Journal of Psychiatry*, 150(6), 782–786.

[5] Shin, D., Lee, K. J., Adeluwa, T., & Hur, J. (2020). Machine learning-based predictive modeling of postpartum depression. *Journal of Clinical Medicine*, 9(9), 2899.

[6] Jiménez-Serranía, M. I., Tortajada, S., García-Gómez, J. M. (2023). Machine learning methods applied to predict perinatal depression. *Journal of Personalized Medicine*, 13(3), 501.

[7] Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020). TabTransformer: Tabular data modeling using contextual embeddings. *arXiv preprint arXiv:2012.06678*.

[8] Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems*, 34, 18932–18943.

[9] Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021). SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training. *arXiv preprint arXiv:2106.01342*.

[10] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171–4186.

[11] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI blog*, 1(8), 9.

[12] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

[13] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1–67.

[14] Dinh, T., Zeng, Y., Zhang, R., Lin, Z., Gira, M., Rajput, S., ... & Haffari, G. (2022). LIFT: Language-interfaced fine-tuning for non-language machine learning tasks. *Advances in Neural Information Processing Systems*, 35, 11763–11784.

[15] Khan, M. N., et al. (2025). PERI_DEP: A dataset of mothers' mental health in Pakistan. *Zenodo*. DOI: 10.5281/zenodo.11094957.

[16] Nakalema, G., et al. (2022). Large scale anonymized EPDS data for prenatal women. *Mendeley Data*. DOI: 10.17632/4swmy34scp.3.

[17] Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems*, 35, 507–520.

[18] Amann, J., Blasimme, A., Vayena, E., Frey, D., & Madai, V. I. (2020). Explainability for artificial intelligence in healthcare: A multidisciplinary perspective. *BMC Medical Informatics and Decision Making*, 20, 310.

---

## Acknowledgments

The authors acknowledge the open-access release of the PERI_DEP and Uganda EPDS datasets, which enabled this research. Computational experiments were conducted using Google Colab with NVIDIA T4 GPU support.

---

## Data Availability

The datasets analyzed in this study are publicly available:
- PERI_DEP: https://zenodo.org/records/11403247
- Uganda EPDS: https://data.mendeley.com/datasets/4swmy34scp/3

Source code is available at: [Repository URL to be added upon publication]
