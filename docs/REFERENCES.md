# 선행 연구와 공개 자산

링크는 논문 원문, 공식 코드 또는 공식 모델 카드만 사용했다. 별도 표기가 없는 성능은 이 프로젝트에서 재현했다고 주장하지 않는다.

## 대회

- [DACON 236749 — 딥보이스 범죄 대응을 위한 AI 탐지 모델 경진대회](https://dacon.io/competitions/official/236749/overview/description)
- [평가 산식](https://dacon.io/competitions/official/236749/overview/evaluation)
- [규칙](https://dacon.io/competitions/official/236749/overview/rules)

## 음성 진위

- W. Ge, X. Wang, X. Liu, J. Yamagishi. [Post-training for Deepfake Speech Detection](https://arxiv.org/abs/2506.21090), 2025.
  - [공식 코드](https://github.com/nii-yamagishilab/AntiDeepfake)
  - [MMS-300M 공식 모델 카드](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake)
- Speech Arena. [DF Arena 1B V1 model card](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1).
- Y. Li et al. [SONAR: Spectral-Contrastive Audio Residuals for Generalizable Deepfake Detection](https://arxiv.org/abs/2511.21325), 2025. 검토했으나 최종 model에는 미포함.

## 음악 진위와 표현

- M. A. Rahman et al. [SONICS: Synthetic Or Not — Identifying Counterfeit Songs](https://arxiv.org/abs/2408.14080), 2024.
  - [공식 코드](https://github.com/awsaf49/sonics)
- Y. Li et al. [MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training](https://arxiv.org/abs/2306.00107), 2023.
  - [MERT-v1-95M 모델 카드](https://huggingface.co/m-a-p/MERT-v1-95M)
- H. Oh. [ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics](https://arxiv.org/abs/2604.16254), 2026.
  - [공식 모델 카드](https://huggingface.co/intrect/artifactnet)
- D. Afchar, G. Meseguer-Brocal, K. Akesbi, R. Hennequin. [A Fourier Explanation of AI-music Artifacts](https://arxiv.org/abs/2506.19108), ISMIR 2025.
  - [Deezer 연구 코드](https://github.com/deezer/ismir25-ai-music-detector)
  - [독립 lofcz 가중치와 구현](https://huggingface.co/lofcz/ai-music-detector)
  - [ArtifactBench](https://github.com/Intrect-io/artifactbench)

## 분리와 문맥

- S. Rouard, F. Massa, A. Défossez. [Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553), ICASSP 2023.
  - [HTDemucs 공식 코드](https://github.com/facebookresearch/demucs)
- Q. Kong et al. [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211), 2019.
  - [공식 코드](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- H. Tak et al. [Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2Vec 2.0 and Data Augmentation](https://arxiv.org/abs/2202.12233), 2022.
  - [RawBoost 공식 코드](https://github.com/TakHemlata/RawBoost-antispoofing)

## 외부 데이터

- [MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny)
- [FakeMusicCaps](https://zenodo.org/records/15063698)
- [SONICS dataset](https://huggingface.co/datasets/awsaf49/sonics)
- [FMA](https://github.com/mdeff/fma)
- [AIME](https://huggingface.co/datasets/disco-eth/AIME)
- [VocalSet](https://zenodo.org/records/1193957)

## 저장소에 포함하지 않은 검토 대상

DeepFense, All-Type-ADD, MusicDET, SOFIA, MoM-CLAM, FST, CtrSVDD baseline도 검토했다. 실행 가능한 checkpoint의 라이선스 불명확성, 체크포인트 부재/오류 보고, 자원 비용, 또는 동일 dev에서의 추가 가치 부족 때문에 최종 구성에 넣지 않았다. “논문에서 높은 점수”와 “이 대회에 합법적으로 재현 가능한 개선”을 구분했다.

