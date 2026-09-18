# Response to the Independent JIM-Style Review

**Manuscript:** *Safe Gated Residual Neural Fusion with Group Split-Conformal Uncertainty Quantification for Milling Surface Roughness*  
**Revision date:** 16 September 2026  
**Decision rule:** Suggestions were adopted only after checking them against the frozen experiment, manuscript assets, and the official *Journal of Intelligent Manufacturing* submission guidance. No new training, tuning, fold/seed removal, or result-driven model change was performed.

## Major comments

### M1. Innovation boundary

**Response: accepted.** The Introduction, Related Work, and Conclusions now state directly that the contribution is an application-level integration and evaluation protocol, not a new gating primitive or conformal theorem. The revised text distinguishes the established components from the paper’s technical increment: an explicit process fallback, group-cross-fitted residual correction, group-level harm assessment, end-to-end group isolation, and simultaneous new-group calibration in one weak-increment vibration setting.

### M2. Data scale and external validity

**Response: accepted with a factual correction.** The review referred to 2,212 machining groups; the verified dataset contains **212** independent machining groups, 586 regions, and 1,758 repeated readings. Section 8.1 now emphasizes that the effective independent sample size is 212, that 189 observed process combinations correspond to only about 1.12 independent groups per observed combination on average, and that broad grid occupancy is not the same as within-combination replication. Limitations for transfer across material, machine, cutter geometry, wear stage, mounting, coolant regime, and process range are now explicit.

### M3. Original versus corrective evidence

**Response: accepted.** The Abstract, Results, and Conclusions now separate two evidence tiers. The original Phase B analysis is the primary evidence for point prediction and material negative transfer. Only the frozen post-audit corrective analysis is used for uncertainty quantification. Table 3 is split into a historical-primary panel and a reduced-proper-training corrective panel so the latter cannot be mistaken for a replacement point-prediction result.

### M4. Why heteroscedastic scaling underperformed

**Response: partially accepted.** A post-audit exploratory diagnostic was added using only already frozen outer-held-out predictions; no model was refitted. The heteroscedastic scale had weak association with realized error, and 21 region–seed predictions had σ ≤ 0.01 μm, producing an extreme normalized-score tail. These observations explain how conformal calibration could recover coverage while retaining poor efficiency, but they do not identify a unique cause. The manuscript explicitly rules out “overfitting calibration labels” because calibration groups were excluded from fitting. Proper-training overfit, insufficient repeat information, and unstable scale ranking remain hypotheses. The suggested search for favorable process subgroups was not performed because it was not pre-specified and the process combinations have little independent replication.

## Minor comments

### m1. The 0.01 μm threshold and group weights

**Response: clarified without retrospective justification.** The manuscript now states that 0.01 μm was a frozen operational deadband, not a value derived from instrument accuracy, a universal manufacturing tolerance, a power calculation, or an economic loss function. The weighting (w_{gi}=1/n_g) is also identified as deterministic normalization for an equal-group estimand, not inverse-probability weighting; an unrelated sampling-weight citation was therefore not added.

### m2. Statistical precision

**Response: accepted with a correction to the proposed interpretation.** The original G1–P1 95% interval for MAE improvement was [-0.003133, 0.002817] μm. Its upper bound therefore excludes an average improvement as large as 0.005 μm under the adopted resampling analysis; it does not show that the sample is too small to detect exactly that effect. The revised paper reports this precision statement while avoiding a formal equivalence claim because no engineering equivalence margin was registered.

### m3. References

**Response: accepted in targeted form.** The bibliography increased from 14 to 26 verified sources. Added coverage includes machining roughness reviews, milling vibration and chatter physics, Monte Carlo dropout, deep ensembles, aleatoric/epistemic uncertainty, conformal regression, covariate shift, nonexchangeability, and proper scoring rules. The 2026 Gao et al. article was verified as formally published in *Machines* with DOI 10.3390/machines14080940. Citations were converted to the author–year format requested by the journal guidance, and the reference list was alphabetized.

### m4. Figures and tables

**Response: partially accepted.** Tables 3–5 are now embedded directly in both the English manuscript and the Chinese review version; the new exploratory Table 6 is embedded as well. Figure 3 already contained an identity diagonal, so no duplicate line was added. A “confidence band” around the perfect-prediction line was not added because that line is a deterministic reference, not an estimated regression curve with sampling uncertainty.

### m5. Abstract length

**Response: accepted.** Although the previous abstract was about 288 words rather than approximately 400, the official journal guidance specifies 150–250 words. The revised English abstract is within that range, and the keyword list was reduced from eight to six to meet the journal’s 4–6 keyword guidance.

## Writing and presentation

**Response: accepted.** The evidence hierarchy and innovation boundary were made explicit, several long passages were tightened, and the internal Chinese-review note was removed from the manuscript. Both language versions now use the same results hierarchy, table content, limitations, and exploratory-status labels.
