# Azure OpenAI GPT Models Latency Comparison

Azure AI Foundry Agent SDK를 사용하여 배포된 모든 GPT 모델의 레이턴시를 비교하는 테스트 도구입니다.
`.env`에 설정된 각 **5.x 모델**에 대해 설정 가능한 `reasoning_effort` 값을 다양하게 적용하여 지연시간을 측정합니다.

**주요 장점:**
- ✅ **API version 관리 불필요** - SDK가 자동으로 처리
- ✅ DefaultAzureCredential로 간편한 인증
- ✅ Reasoning 모델 네이티브 지원
- ✅ **`.env` 기반 자동 구성** - `GPT_*_DEPLOYMENT_NAME` 항목을 자동 감지

## 지원 모델

`.env`의 `GPT_*_DEPLOYMENT_NAME` 항목을 자동으로 읽어 테스트 대상을 구성합니다.

### 비추론(Non-reasoning) 모델
- **gpt-4o**, **gpt-4.1**: 표준 Chat Completions API (reasoning_effort 미적용)

### 추론(Reasoning) 5.x 모델
- **gpt-5.1**, **gpt-5.2**, **gpt-5.4**, **gpt-5.4-mini**, **gpt-5.4-nano**
- 각 모델을 `reasoning_effort` 레벨별로 한 번씩 테스트합니다.

## 주요 기능

- ✅ **5.x reasoning effort 스윕**: 각 5.x 모델 × (`none`/`minimal`/`low`/`medium`/`high`) 조합별 지연시간 측정
- ✅ **GPT-4.x 모델**: 표준 Chat Completions API로 비교 기준 제공
- ✅ **다양한 프롬프트**: 간단한 질문부터 복잡한 작업까지 반복 테스트
- ✅ **상세 분석**: 레이턴시(ms), 토큰 사용량, **reasoning 토큰** 통계
- ✅ **CSV 결과**: `reasoning_effort`, `reasoning_tokens` 컬럼 포함 상세 결과 저장

## Reasoning effort (5.x 모델)

GPT-5.x 시리즈는 **adaptive reasoning**을 지원하며, `reasoning_effort` 파라미터로 제어합니다
(추론 강도 높은 순으로 실행):

- `high`: 복잡한 작업으로 최대 정확도가 필요한 경우 (최고 지연)
- `medium`: 일반적인 작업
- `low`: 빠른 응답이 필요하지만 약간의 추론이 필요한 경우
- `minimal`: 최소 추론
- `none`: 추론 비활성화 (최저 지연, 5.1 이상)

측정할 effort 목록은 `.env`의 `REASONING_EFFORTS`로 조정할 수 있습니다(쉼표 구분). 미설정 시 기본값은
`high,medium,low,minimal,none` 입니다. **추론 강도가 높은 effort부터 순서대로 실행**되며, 입력 순서와
무관하게 항상 `high → medium → low → minimal → none` 순으로 정렬됩니다. 특정 모델이 지원하지 않는
effort 값은 오류로 기록되어 결과의 `ERRORS` 섹션과 CSV에 함께 표시됩니다.

**참고**: GPT-5.x 모델은 `temperature`, `top_p`, `logprobs` 파라미터를 지원하지 않습니다.

## 설치

```bash
pip install -r requirements.txt
```

## 설정

1. `.env.example` 파일을 `.env`로 복사:
```bash
cp .env.example .env
```

2. `.env` 파일에 Azure AI 정보 입력:
```
# Azure AI Foundry Project Endpoint
AZURE_AI_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project

# 모델 배포 이름 (Azure AI Foundry에 배포된 이름과 동일하게 설정)
# GPT-4 시리즈 (비추론)
GPT_4o_DEPLOYMENT_NAME=gpt-4o
GPT_41_DEPLOYMENT_NAME=gpt-4.1

# GPT-5.x 시리즈 (추론 - reasoning_effort 스윕 대상)
GPT_51_DEPLOYMENT_NAME=gpt-5.1
GPT_52_DEPLOYMENT_NAME=gpt-5.2
GPT_54_DEPLOYMENT_NAME=gpt-5.4
GPT_54_MINI_DEPLOYMENT_NAME=gpt-5.4-mini
GPT_54_NANO_DEPLOYMENT_NAME=gpt-5.4-nano

# (선택) 측정할 reasoning effort 레벨 - 미설정 시 아래 기본값 사용 (추론 강도 높은 순)
REASONING_EFFORTS=high,medium,low,minimal,none
```

> `GPT_*_DEPLOYMENT_NAME` 형식의 항목은 자동으로 감지됩니다. `gpt-5`가 포함된 배포 이름은
> 추론 모델로 간주되어 `REASONING_EFFORTS`의 각 값으로 테스트되고, 그 외 모델은 그대로 1회 테스트됩니다.

3. Azure CLI로 로그인 (DefaultAzureCredential 사용):
```bash
az login
```

**참고**: API 키 없이 `DefaultAzureCredential`로 인증합니다. Azure CLI 로그인이 필요합니다.

## 실행

```bash
python test_latency.py
```

## 결과

- 콘솔에 실시간 테스트 결과 출력 (모델 + reasoning effort별)
- 모델/effort별 평균/최소/최대 레이턴시 분석
- 토큰 및 reasoning 토큰 사용량 통계
- `latency_results.csv` 파일로 상세 결과 저장 (`reasoning_effort`, `reasoning_tokens` 컬럼 포함)

## 참고 문서

- [GPT-5.4 API 문서](https://platform.openai.com/docs/guides/latest-model)
