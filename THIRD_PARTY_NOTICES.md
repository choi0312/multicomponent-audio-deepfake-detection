# Third-party notices

이 저장소는 학습 음원과 외부 모델 가중치를 포함하지 않습니다. 아래 자산을 전체 파이프라인에서 사용할 때는 원 라이선스와 이용 범위를 직접 확인해야 합니다.

| Component | Upstream | Declared terms relevant to this project |
|---|---|---|
| HTDemucs | https://github.com/facebookresearch/demucs | MIT code; verify checkpoint terms |
| PANNs | https://github.com/qiuqiangkong/audioset_tagging_cnn | Upstream repository terms |
| SONICS | https://github.com/awsaf49/sonics | MIT code/model release; dataset terms separate |
| MMS-300M AntiDeepfake | https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake | CC BY-NC-SA 4.0 weights; BSD-3-Clause upstream code |
| MERT-v1-95M | https://huggingface.co/m-a-p/MERT-v1-95M | CC BY-NC 4.0 weights |
| ArtifactNet | https://huggingface.co/intrect/artifactnet | CC BY-NC 4.0; read model-card patent notice |
| Deezer Fourier research method | https://github.com/deezer/ismir25-ai-music-detector | CC BY-NC 4.0 research code |
| lofcz Fourier checkpoint | https://huggingface.co/lofcz/ai-music-detector | Model card declares MIT; independent weights, not Deezer official weights |
| ArtifactBench adapter | https://github.com/Intrect-io/artifactbench | MIT adapter; underlying asset terms remain applicable |

`src/fourier_detector.py`는 위 논문과 공개 구현을 명시적으로 인용하는 교육·비상업 연구용 재구성입니다. 이 저장소 전체에 하나의 허가가 제3자 가중치나 데이터의 조건을 덮어쓴다는 의미로 해석해서는 안 됩니다.

데이터셋별 저작권과 개별 오디오 라이선스도 각각 적용됩니다. 특히 FMA는 트랙별 라이선스가 다르며, 본 프로젝트는 No-Derivatives 트랙을 제외했습니다. AIME의 generated audio, VocalSet, MUSIC8K 등도 원 배포 페이지의 attribution 조건을 따라야 합니다.

저장소 자체에는 별도 오픈소스 라이선스를 부여하지 않았습니다. 재사용 라이선스를 추가하려면 작성자 소유 코드와 제3자 파생 부분의 경계를 먼저 법률적으로 검토해야 합니다.

