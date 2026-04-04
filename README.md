# Azure OpenAI GPT Models Latency Comparison

Azure AI Foundry Agent SDK를 사용하여 배포된 모든 GPT 모델의 레이턴시를 비교하는 테스트 도구입니다.

**주요 장점:**
- ✅ **API version 관리 불필요** - SDK가 자동으로 처리
- ✅ DefaultAzureCredential로 간편한 인증
- ✅ Reasoning 모델 네이티브 지원

## 지원 모델

### GPT-4 시리즈
- **gpt-4.1**: 표준 Chat Completions API

### GPT-5.4 시리즈 (Latest)
- **gpt-5.4**: Adaptive reasoning 지원
- **gpt-5.4-mini**: Adaptive reasoning 지원

## 주요 기능

- ✅ **GPT-5.4 adaptive reasoning**: 다양한 `reasoning_effort` 설정으로 테스트
- ✅ **GPT-4.x 모델**: 표준 Chat Completions API로 비교
- ✅ **다양한 프롬프트**: 간단한 질문부터 복잡한 작업까지 반복 테스트
- ✅ **상세 분석**: 레이턴시(ms), 토큰 사용량 통계
- ✅ **CSV 결과**: 상세 결과를 CSV 파일로 저장

## GPT-5.4 특징

GPT-5.4는 **adaptive reasoning**을 지원하며, `reasoning.effort` 파라미터로 제어합니다:

- `minimal`: 최소 추론 (최저 지연)
- `low`: 빠른 응답이 필요하지만 약간의 추론이 필요한 경우
- `medium`: 일반적인 작업
- `high`: 복잡한 작업으로 최대 정확도가 필요한 경우

**참고**: GPT-5.4 모델은 `temperature`, `top_p`, `logprobs` 파라미터를 지원하지 않습니다.

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
# GPT-4 시리즈
GPT_41_DEPLOYMENT_NAME=gpt-4.1

# GPT-5.4 시리즈
GPT_54_DEPLOYMENT_NAME=gpt-5.4
GPT_54_MINI_DEPLOYMENT_NAME=gpt-5.4-mini
```

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

- 콘솔에 실시간 테스트 결과 출력
- 모델별 평균/최소/최대 레이턴시 분석
- 토큰 사용량 통계
- `latency_results.csv` 파일로 상세 결과 저장

## 참고 문서

- [GPT-5.4 API 문서](https://platform.openai.com/docs/guides/latest-model)
