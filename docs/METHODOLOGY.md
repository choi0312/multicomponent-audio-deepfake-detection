# 방법론 상세

## 목표 함수에서 시작한 설계

대회는 File EER, Voice EER, Music EER와 두 성분 존재 AUC를 함께 평가한다. Voice/Music EER은 해당 성분이 존재하는 파일에서만 계산되므로, `어떤 성분이 있는가`와 `그 성분이 가짜인가`를 구분해야 한다.

최종 v3의 목표는 이미 높았던 CPS를 고정하고 ADS만 개선하는 것이었다. 따라서 출력 벡터를 다음처럼 다뤘다.

```text
[file_fake, voice_fake, music_fake, voice_present, music_present]
  └──────────── new ADS heads ────────────┘  └── exact v2 copy ──┘
```

이 분리는 모델 선택뿐 아니라 코드 수준에서 검사한다. 최종 후처리 함수는 마지막 두 열을 v2에서 대입하며, 단위 테스트와 실제 파형 비교에서 bit-exact 여부를 확인한다.

## 입력과 조건부 분리

모든 오디오는 mono 16 kHz float32로 변환한다. PANNs Cnn14에서 speech, singing, voice, music 문맥을 추출한다. 기존 존재 추정이나 PANNs가 성분을 나타낼 때 HTDemucs를 실행하고, 그렇지 않으면 원본을 재사용한다.

HTDemucs는 44.1 kHz에서 실행하고 vocals stem을 16 kHz로 되돌린다. 반주는 개별 drums/bass/other stem의 합이 아니라 `입력 - 보컬`로 정의한다. 이 선택은 재구성 오차로 원본 에너지가 사라지는 것을 막는다.

훈련용 합성 장면에도 같은 분리기를 적용했다. head는 이상적인 원천이 아니라 실제 inference와 같은 누출·잔향·분리 artifact를 본다.

## 음성 전문가

### DF Arena 1B

대회 baseline의 음성 SSL detector를 유지했다. 실행 자원에 맞추기 위해 계층 표현을 한 번에 쌓지 않고 누적 가중합으로 계산하고, 위치 임베딩 초기화를 벡터화했다. 변환 전후 출력 parity를 별도로 확인했다.

### MMS-300M AntiDeepfake

AntiDeepfake는 대규모 다국어 진짜 음성 및 다양한 artifact를 이용해 SSL encoder를 deepfake detection에 맞게 post-train한다. 원본과 vocal stem의 최대 3개 창에서 점수를 추출한다.

초기 체크포인트 경량화에서 FP16 오차가 관측되어 최종 detection 경로는 FP32를 유지했다. inference에 사용하지 않는 pretraining-only parameter만 제외했다.

### SONAR: 검토 후 제외

SONAR의 원신호·고주파 잔차 이중 encoder와 spectral residual은 기존 두 음성 detector에 상보적일 가능성이 있었다. 공식 FP32 모델을 변환하고 별도 구현과 수치 비교했지만, 동일한 480후보 프로토콜에서 ArtifactNet+Fourier 조합보다 최종 목적함수가 좋지 않아 제출 구성에서는 제외했다.

이 결과는 SONAR 자체가 나쁘다는 뜻이 아니다. 현재 외부 데이터, 계산 예산, 다른 전문가와의 중복 아래에서 추가 가치가 선택 기준을 넘지 못했다는 뜻이다.

## 음악 전문가

### SONICS SpecTTTra

SONICS는 singing-voice deepfake에 한정된 기존 데이터의 범위를 넘어, 보컬·반주·가사·스타일 전체가 생성될 수 있는 end-to-end synthetic song을 다룬다. SpecTTTra는 긴 곡의 시간 의존성을 효율적으로 모델링한다.

이 프로젝트에서는 baseline 5초 encoder를 외부 음악/혼합 장면에 적응시키고, 원본 및 분리 반주 창에서 점수와 표현을 얻었다. 깨끗한 음악 적응 branch와 음성 간섭이 있는 혼합 적응 branch를 비교했다.

### MERT-v1-95M

MERT는 RVQ-VAE 음향 교사와 CQT 음악 교사의 pseudo-label을 사용하는 self-supervised 음악 표현이다. encoder는 동결하고 여러 계층의 표현에 작은 classifier를 학습했다. 데이터가 제한된 상황에서 전체 encoder fine-tuning보다 안정적인 보조 신호로 사용했다.

### ArtifactNet zero-energy extension

ArtifactNet은 44.1 kHz 4초 구간에서 UNet residual과 HPSS 기반 7채널 포렌식 특징을 사용한다. 공개 ONNX가 일부 무음/희소 입력에서 NaN을 만드는 원인을 추적한 결과, HPSS 내부의 정확한 영분모가 원인이었다.

수정은 분모가 정확히 0인 위치에서만 성분을 0으로 정의한다. 양의 분모 경로와 학습 파라미터는 유지했다.

- CPU 73개 창 전체 유한
- 원본이 정의되는 58개 창 출력 차이 0
- GPU 77개 창의 CPU 대비 최대 확률 차이 0.000403 미만

원본과 분리 반주에서 최대 3개의 결정적 창을 보고 중앙값을 기본 점수로 사용한다. 이는 upstream의 긴 곡 분석 정책을 짧은 대회 파일에 맞게 제한한 것이다.

### Fourier fakeprint

Afchar et al.은 생성 모델의 deconvolution 구조가 학습 데이터나 개별 가중치와 별개로 작은 주기적 주파수 peak를 만들 수 있음을 설명한다. 사용한 분류기는 해당 아이디어를 구현한 독립 lofcz checkpoint이며 Deezer 공식 production 모델이 아니다.

파이프라인은 16 kHz audio의 8,192-point power spectrum에서 1–8 kHz 구간 3,585개 bin을 취한다. 하위 envelope를 제거하고 남은 peak를 0–5 dB로 제한한 뒤 파일 내부 최대값으로 정규화한다.

두 공개 구현의 차이를 보존했다.

1. `minimum_filter1d(size=10, mode=nearest)`
2. 이동 창 최솟값을 이은 quadratic lower envelope

두 logit을 원본과 반주에서 따로 추출한다. 특정 코덱/생성기에서 한 방식의 편향이 나타날 수 있어 사전에 평균하지 않았다.

## 문맥 gate와 성분 head

고정 PANNs logit에서 다음 비율을 만든다.

```text
singing_context = sigmoid(singing) /
                  (sigmoid(singing) + sigmoid(speech) + 0.02)

speech_context  = sigmoid(speech) /
                  (sigmoid(speech) + sigmoid(singing) + 0.05)
```

Voice head는 singing context를, Music head는 speech context를 사용한다. 각 점수는 context/ordinary 두 항으로 나뉜다. Voice head에는 음악 전문가를 넣지 않고, Music head에는 음성 전문가를 넣지 않아 명시적 cross-component shortcut을 줄였다.

각 head는 weighted logistic regression이다. 계수는 `[0, 3]`, 이전 모델 anchor는 `[0, 0.5]`로 제한한다. 이전 모델 특징에는 4배 L2 penalty를 부과한다. 정규화 평균과 scale은 train에서만 계산하고 모델 파일에 저장한다.

선택된 정규화 강도는 Voice `0.03`, Music `0.003`, File `0.03`이다. component logit temperature는 `2.0`이다.

## 파일 head

성분 진위와 존재 확률로 noisy-or 형태의 union을 만든다.

```text
union = 1 - (1 - vp × voice_fake) × (1 - mp × music_fake)
```

learned file head는 `union`, `vp×voice_logit`, `mp×music_logit`, 이전 file logit 네 값을 사용한다. 이 head 역시 비음수 계수로 학습한다.

## 학습 데이터와 장면 합성

선택 외부 음원은 MLAAD-tiny, FakeMusicCaps, SONICS, FMA, AIME, VocalSet에서 왔다. 원본과 처리 사본을 함께 센 보수적 합계는 4.9069 GB다. 데이터나 manifest에 있는 라이선스를 무시하고 전체 corpus를 재배포하지 않는다.

장면은 다음을 포함한다.

- real/fake voice × real/fake music 네 조합
- voice-only, music-only
- 두 성분의 순차 등장
- 상대 음량, 채널, telephone-band, μ-law, noise, gain 변화
- solo vocal과 vocal+accompaniment control

같은 원 발화, 프롬프트, 가수, 아티스트는 split을 넘지 않게 그룹 단위로 분리했다.

## 코덱 학습

8개 joint label 조합마다 train 40개, dev 20개 부모를 고정 seed hash로 선택했다. 각 부모에 MP3/AAC/G.711 view를 만들었다. train 960, dev 480개의 변환 행이 추가된다.

부모와 view가 공유하는 총 학습 가중치는 1이다. source/label group도 균형화해 큰 source가 loss를 지배하지 않도록 했다.

## 후보 탐색

전문가 설정 10개는 baseline, SONAR, ArtifactNet, 두 Fourier 방식, Artifact+Fourier, full 조합, temporal max, 이전 점수 제외 등을 포함한다. 각 설정에 clean/codec training을 적용한다.

융합은 다음 축을 조합한다.

- feature mode: pure / gated / stem-only
- component temperature: 1 / 2
- file head: union / max / soft-gated union / learned

총 480개를 동일한 dev에서 완전 평가했다. 결측 cache나 불완전 실험은 최종 순위에 넣지 않았다.

## 선택 기준과 voice shrinkage

주요 집단 EER 평균만 최소화하면 작은 domain이 무시된다. 세부집단 최악값만 최소화하면 표본 잡음에 과민하다. 그래서 평균과 신뢰 가능한 하위집단의 90백분위를 반씩 결합했다.

```text
robust_task_error = 0.5 × mean(major EER)
                  + 0.5 × p90(subgroup EER)
```

각 class가 10개 이상인 집단만 선택 기준에 들어간다. 최종 ADS 선택 오류는 공식 component weight와 같은 `0.5/0.2/0.3`을 사용한다.

1위 후보에서 vocal fake-accompaniment의 voice regression이 발견돼, voice 확률만 v2와 혼합했다. 동일 dev에서 다섯 alpha를 추가 비교했고 `alpha=0.5`를 선택했다. 이 과정은 별도 holdout이 아니므로 결과 보고서에서 post-hoc 선택으로 명시한다.

## 파일 독립 추론

모든 window aggregation, Fourier normalization, separation, expert scoring은 한 파일 내부에서 끝난다. 대회 test 파일들의 순위·평균·분위수·예측 분포를 공유하지 않는다. 입력 파일 순서를 뒤집은 실제 패키지 검사에서 출력 차이가 0이었다.

