# Research and references

This file lists every paper PulseGuard draws on, what each contributed, and how much of it was actually read.
Reference numbers (`[R1]`, `[R2]`, ...) match the README.

**Reading status** (stated so you know how far to trust each summary):

- **Read**: full text of the relevant sections was read for this project.
- **Verified**: title, authors and venue were confirmed against a public listing, but the paper was not read in full.
- **From memory**: cited from general knowledge of the field. Check volume and page details before citing formally.

## Foundation: the survey this project starts from

| # | Reference | Status | Contribution to PulseGuard |
|---|---|---|---|
| R1 | K. S. Kammari, M. Khanna, Y. L. Annambhotla, K. D. Anuradha, P. N. S. Namrata, K. V. Nagaraja, T. V. Smitha, T. Jan, G. R. Naik, "A Comprehensive Review of Deepfake Detection Techniques Utilizing Remote Photoplethysmography," *IEEE Access*, vol. 13, pp. 183557+, 2025. DOI 10.1109/ACCESS.2025.3624284. PDF in this repo. | **Read** (introduction, literature review, challenges, future directions, conclusion; the per-method sections were skimmed) | Defines the problem and the weaknesses listed in the README. Source of these design ideas: illumination correction, quality gating, cross-modal consistency, region-wise coherence, privacy-preserving processing, and moving past raw signal periodicity because generators can imitate rPPG. |

Points from R1 worth knowing:

- Section VII-C states that GANs and diffusion models can produce rPPG-like signals, which "questions whether rPPG
  provides a unique advantage" unless detectors use more than periodicity. This is why PulseGuard treats rPPG as
  one modality and fuses it with others.
- The survey reports 95% accuracy in the future-work section and "up to 92%" for DeepRhythm in the conclusion,
  without a single clear benchmark. Treat its accuracy numbers as indicative only.
- Its conclusion names **skin colour variation and lighting** as the two main obstacles. PulseGuard is built around those.

## Pulse extraction (lighting robustness)

| # | Reference | Status | How it is used |
|---|---|---|---|
| R2 | G. de Haan and V. Jeanne, "Robust pulse rate from chrominance-based rPPG," *IEEE Trans. Biomed. Eng.*, 60(10):2878-2886, 2013. | From memory | CHROM algorithm in `rppg.py`. The SNR definition (power near the peak and first harmonic over the rest of the band) in `quality.py` follows this paper. |
| R3 | W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan, "Algorithmic principles of remote PPG," *IEEE Trans. Biomed. Eng.*, 64(7):1479-1491, 2017. | From memory | POS algorithm in `rppg.py`. Default method. Projects colour onto a plane orthogonal to the skin tone so common brightness changes cancel. |
| R4 | W. Verkruysse, L. O. Svaasand, J. S. Nelson, "Remote plethysmographic imaging using ambient light," *Optics Express*, 16(26):21434-21445, 2008. | From memory | Origin of the green-channel signal. Kept only as the baseline POS and CHROM are compared against. |
| R10 | X. Li, J. Chen, G. Zhao, M. Pietikäinen, "Remote heart rate measurement from face videos under realistic situations," *CVPR* 2014. [PDF](https://openaccess.thecvf.com/content_cvpr_2014/papers/Li_Remote_Heart_Rate_2014_CVPR_paper.pdf) | Verified | Idea of using a background region to rectify illumination, implemented as `--ref-correct` and `illumination_correct()`. Their work targets exactly the motion and illumination failures noted in R1. |

## rPPG-based deepfake detectors (prior art)

| # | Reference | Status | Relevance |
|---|---|---|---|
| R5 | U. A. Ciftci, I. Demir, L. Yin, "FakeCatcher: Detection of synthetic portrait videos using biological signals," *IEEE TPAMI*, 2020 (early access). | From memory | Established that biological signals separate real and synthetic portrait videos. Motivates using pulse features and cross-region consistency. |
| R6 | H. Qi et al., "DeepRhythm: Exposing DeepFakes with attentional visual heartbeat rhythms," *ACM Multimedia*, 2020. | From memory | Heart-rhythm-based detector discussed in R1. Reference point for what a learned rPPG detector looks like. |
| R7 | J. Hernandez-Ortega, R. Tolosana, J. Fierrez, A. Morales, "DeepFakesON-Phys: DeepFakes detection based on heart rate estimation," arXiv:2010.00400, 2020. [arXiv](https://arxiv.org/abs/2010.00400) | Verified | Learned detector using a convolutional attention network. Reported strong results on Celeb-DF v2 and DFDC. |
| R14 | "Physiological Signals as a Forensic Modality for Talking-Face Deepfake Detection," arXiv:2607.21776, 2026 (preprint, not peer reviewed). [arXiv](https://arxiv.org/abs/2607.21776) | Verified from abstract only | Reports a reproduction of DeepFakesON-Phys falling from AUC 0.999 on older face-swap data to 0.622 on talking-face fakes. Supports the claim that rPPG detectors generalise poorly and motivates multimodal fusion and cross-dataset testing. |

## Skin tone, fairness and bias

| # | Reference | Status | How it is used |
|---|---|---|---|
| R8 | E. M. Nowara, T. K. Marks, H. Mansour, A. Veeraraghavan, "A meta-analysis of the impact of skin tone and gender on non-contact photoplethysmography measurements," *CVPR Workshops*, 2020. | From memory | Evidence that skin tone affects rPPG signal quality. Basis for the skin-tone reliability factor and for per-group reporting. |
| R9 | A. Dasari, S. K. A. Prakash, L. A. Jeni, C. S. Tucker, "Evaluation of biases in remote photoplethysmography methods," *npj Digital Medicine*, 4:91, 2021. [Article](https://www.nature.com/articles/s41746-021-00462-z) | Verified | Found that rPPG methods show biases across demographic groups and environmental conditions. Motivates fairness tests as acceptance criteria. |
| R17 | "Demographic bias in public remote photoplethysmography datasets," *npj Digital Medicine*, 2025. [Article](https://www.nature.com/articles/s41746-025-01973-9) | Verified from abstract only | Audits 100 rPPG studies and reports under-representation of darker skin tones and gender imbalance. Explains why a model trained on public rPPG data cannot be assumed fair. |
| R11 | M. Chardon, J. Cretois, C. Hourseau, "Skin colour typology and suntanning pathways," *Int. J. Cosmetic Science*, 13(4):191-208, 1991. | From memory | Individual Typology Angle (ITA), used as a coarse skin-tone proxy from pixel colour. Lighting-confounded, so only a soft flag. |
| R12 | T. B. Fitzpatrick, "The validity and practicality of sun-reactive skin types I through VI," *Arch. Dermatology*, 124(6):869-871, 1988. | From memory | Standard skin-type scale. The evaluation plan requires per-type reporting using Fitzpatrick labels where datasets provide them. |
| R13 | J. Buolamwini and T. Gebru, "Gender Shades: Intersectional accuracy disparities in commercial gender classification," *PMLR* 81:77-91, 2018. | From memory | Methodological precedent for reporting accuracy per subgroup instead of one overall number. |

## Datasets for the evaluation plan (not yet used)

| # | Reference | Status | Role |
|---|---|---|---|
| R15 | A. Rössler et al., "FaceForensics++: Learning to detect manipulated facial images," *ICCV* 2019. | From memory | Deepfake benchmark. |
| R16 | Y. Li et al., "Celeb-DF: A large-scale challenging dataset for DeepFake forensics," *CVPR* 2020. | From memory | Higher-quality fakes, cross-dataset testing. |
| R20 | B. Dolhansky et al., "The DeepFake Detection Challenge (DFDC) dataset," arXiv:2006.07397, 2020. | From memory | Large, diverse fake set. |
| R18 | S. Bobbia et al., "Unsupervised skin tissue segmentation for remote photoplethysmography," *Pattern Recognition Letters*, 124:82-90, 2019 (UBFC-rPPG). | From memory | Real videos with contact pulse ground truth, for measuring pulse accuracy. |
| R19 | X. Niu et al., "VIPL-HR: A multi-modal database for pulse estimation from less-constrained face video," *ACCV* 2018. | From memory | Less-constrained conditions (motion, lighting). Also listed in R1. |

## Provenance (complementary to detection)

| # | Reference | Status | Role |
|---|---|---|---|
| R21 | Coalition for Content Provenance and Authenticity (C2PA), *Technical Specification*. https://c2pa.org | From memory | Signed content credentials let authentic media prove origin, so detection does not have to catch every fake. Roadmap item, not implemented. R1 also points to signature-based verification. |

## Context on harm

| # | Reference | Status | Role |
|---|---|---|---|
| R22 | B. Chesney and D. Citron, "Deep fakes: A looming challenge for privacy, democracy, and national security," *California Law Review*, 107:1753, 2019. (cited in R1) | From memory | Legal and societal framing of the harms the project aims to reduce. |

## How the research maps to design decisions

| Design decision | Evidence |
|---|---|
| Use POS/CHROM instead of the green channel | R2, R3, R4, and our synthetic test where green locked onto a lighting artefact and POS did not |
| Gate on signal quality and capture conditions, abstain when low | R1 section VII (lighting, compression, motion limits); R10 |
| Penalise reliability, not just accuracy, for dark skin, and report per group | R8, R9, R17, R13 |
| Fuse rPPG with other modalities | R1 sections VII-C and VIII-B/C, R14 |
| Cross-region coherence feature | R1 (fakes show uniform or inconsistent rPPG across regions), R5 |
| Cross-dataset evaluation | R1 section VIII-D, R14 |
| Privacy-aware handling | R1 section VIII-F |
| Provenance as a second line of defence | R21, R1 |

## Gaps in this list

- Only R1 was read in depth. Before citing anything formally, check volumes, pages and DOIs for entries marked "From memory".
- No literature on diffusion-era generators or adversarial attacks on rPPG detectors has been reviewed beyond what R1 and R14 say. That is a priority for the next literature pass.
