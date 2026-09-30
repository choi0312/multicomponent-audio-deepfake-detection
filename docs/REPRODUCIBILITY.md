# 재현 범위

## 이 저장소에서 즉시 재현 가능한 것

- 문맥 gate와 feature ordering
- 비음수 weighted logistic head 학습
- robust selection error
- v2 voice shrinkage와 CPS exact pass-through
- codec view 생성 및 부모 공유 가중치
- Fourier 두 lower-envelope 전처리

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 전체 파이프라인에 필요한 외부 자산

| 자산 | 고정 revision | 라이선스/주의 |
|---|---|---|
| DF Arena 1B V1 | `fb6ce85de12c2c5a509d89114adaf827dd75f49f` | 모델 카드 확인 |
| HTDemucs `955717e8` | 공식 Demucs model zoo | MIT code, checkpoint 조건 확인 |
| PANNs Cnn14 | `Cnn14_mAP=0.431.pth` | 공식 저장소 확인 |
| SONICS SpecTTTra 5s | `fb311b279eabd9b80918874f085e3c1ad7e46f4a` | MIT |
| MMS-300M AntiDeepfake | `7928e119e09f616b2ce698892e00ca22dbaa40bf` | CC BY-NC-SA 4.0 |
| MERT-v1-95M | `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5` | CC BY-NC 4.0 |
| ArtifactNet v9.4 | `e915f0dc5962a57536bbe1f78b66adcb48dfae4c` | CC BY-NC 4.0, patent notice 확인 |
| lofcz Fourier detector | `d2180598fed79e3f917e8050a00439982466e5c6` | 모델 카드 MIT 선언 |

원 가중치는 이 저장소에서 재배포하지 않는다. 공개 모델의 revision이 사라지거나 라이선스가 바뀌면 동일한 전체 실행을 보장할 수 없다.

## 데이터 구성

오디오 자체는 포함하지 않는다. 그룹 분할 원칙은 다음과 같다.

- MLAAD: 같은 원 발화 이름을 같은 split으로 유지
- FakeMusicCaps/SONICS/AIME: 같은 prompt/source identity의 변형을 같은 split으로 유지
- FMA: artist 단위 분리, No-Derivatives 트랙 제외
- VocalSet: singer-disjoint split
- Mureka stress subset: fitting 및 checkpoint 선택에서 제외

선택 원본+처리 사본의 보수적 합계는 4,906,853,896 bytes다. 모델 가중치와 numerical cache는 새 오디오 record가 아니므로 이 합계와 분리해 기록했다.

## 전체 훈련 흐름

```text
1. 외부 음원의 라이선스와 source identity를 manifest에 고정한다.
2. group-disjoint train/dev/test split을 만든다.
3. 8개 joint label 조합과 vocal control 장면을 생성한다.
4. inference와 같은 HTDemucs 및 expert로 feature cache를 만든다.
5. 고정 plan의 codec views를 메모리에서 생성한다.
6. 10 × 2 × 24 후보를 train label만으로 fitting한다.
7. 각 class 10개 이상인 dev group으로 robust error를 계산한다.
8. 1위 후보에서 voice alpha 5개를 비교하고 설정을 동결한다.
9. 이전에 본 test/stress는 동결 뒤 진단만 하며 재선택하지 않는다.
10. 최종 package를 새 환경에서 offline 실행하고 CPS/순서/자원을 검사한다.
```

## 중요한 불변조건

1. test batch 평균, 순위, percentile을 사용하지 않는다.
2. 한 파일의 특징은 그 파일 내부 window만으로 계산한다.
3. train normalization을 model config에 저장한다.
4. CPS 두 출력은 같은 파일 v2 output을 그대로 복사한다.
5. codec clean/view는 부모 총 가중치를 공유한다.
6. 작은 dev subgroup은 표시하되 모델 선택에는 쓰지 않는다.

## 실행 환경

최종 검증 환경은 Python 3.11, PyTorch/torchaudio 2.7.1+cu128, CUDA 12.8, ONNX Runtime GPU 1.23.2, NVIDIA L4였다. 전체 submission은 인터넷이 차단된 환경에서 실행했으며 Python socket/DNS 시도 0회를 기록했다.

