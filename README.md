# Azure OpenAI GPT Reasoning Effort Latency Lab

Azure OpenAI에 배포한 GPT 모델을 동일한 프롬프트로 호출하여 **비스트리밍 전체 응답 지연시간**과 토큰 사용량을 비교하는 실습입니다. 표준 `openai` Python SDK의 Azure OpenAI **v1 API**, Microsoft Entra ID 인증, Chat Completions API를 사용합니다.

> 이 도구는 Azure에 존재하는 모든 배포를 자동 조회하지 않습니다. `.env`에 명시한 배포만 테스트합니다.

## 지원 구성

Azure의 배포 이름은 사용자가 자유롭게 정할 수 있으므로, 모델 종류는 배포 이름 값이 아니라 환경 변수 키로 판별합니다.

| 환경 변수 | 모델 | 적용하는 `reasoning_effort` |
| --- | --- | --- |
| `GPT_41_DEPLOYMENT_NAME` | `gpt-4.1` | 미적용 |
| `GPT_4O_DEPLOYMENT_NAME` | `gpt-4o` | 미적용 |
| `GPT_51_DEPLOYMENT_NAME` | `gpt-5.1` | `high`, `medium`, `low`, `none` |
| `GPT_52_DEPLOYMENT_NAME` | `gpt-5.2` | `xhigh`, `high`, `medium`, `low`, `none` |
| `GPT_54_DEPLOYMENT_NAME` | `gpt-5.4` | `xhigh`, `high`, `medium`, `low`, `none` |
| `GPT_54_MINI_DEPLOYMENT_NAME` | `gpt-5.4-mini` | `xhigh`, `high`, `medium`, `low`, `none` |
| `GPT_54_NANO_DEPLOYMENT_NAME` | `gpt-5.4-nano` | `xhigh`, `high`, `medium`, `low`, `none` |
| `GPT_56_SOL_DEPLOYMENT_NAME` | `gpt-5.6-sol` | `xhigh`, `high`, `medium`, `low`, `none` |
| `GPT_56_TERRA_DEPLOYMENT_NAME` | `gpt-5.6-terra` | `xhigh`, `high`, `medium`, `low`, `none` |
| `GPT_56_LUNA_DEPLOYMENT_NAME` | `gpt-5.6-luna` | `xhigh`, `high`, `medium`, `low`, `none` |

GPT-5.6의 Sol은 최고 성능, Terra는 성능과 비용의 균형, Luna는 비용 민감형 대량 워크로드를 위한 모델입니다. `gpt-5.6` 별칭은 Sol로 연결되지만, 비교 결과를 명확히 하기 위해 이 실습은 세 모델 ID를 직접 사용합니다.

`minimal`은 이 실습의 GPT-5.1 이상 모델 조합에 유효하지 않으므로 설정 단계에서 오류로 처리합니다. `xhigh`는 이를 지원하는 모델에서만 실행되며 나머지 모델에서는 자동으로 제외됩니다.

GPT-5.6의 `max` effort와 pro mode는 Responses API 기능입니다. 이 실습은 모든 모델을 동일한 Chat Completions API로 비교하므로 `max`와 pro mode는 포함하지 않습니다.

## 사전 준비

- Python 3.10 이상
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
- Azure OpenAI 또는 Microsoft Foundry 리소스와 하나 이상의 모델 배포
- 실행할 사용자 또는 관리 ID에 대상 리소스 범위의 **Cognitive Services OpenAI User** 역할

로컬 사용자 인증:

```bash
az login
az account show
```

여러 테넌트를 사용한다면 `az login --tenant <tenant-id>`로 로그인 대상을 명시합니다.

## 1. 설치

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. 환경 설정

```bash
cp .env.example .env
```

`.env`에서 엔드포인트 하나와 실제 **배포 이름**을 입력합니다.

```dotenv
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

GPT_41_DEPLOYMENT_NAME=my-gpt-41-deployment
GPT_54_DEPLOYMENT_NAME=my-gpt-54-deployment
```

엔드포인트에는 다음 형식 중 하나를 사용할 수 있습니다.

- Azure OpenAI 리소스: `https://<resource>.openai.azure.com`
- Foundry 리소스: `https://<resource>.services.ai.azure.com`
- Foundry 프로젝트: `https://<resource>.services.ai.azure.com/api/projects/<project>`
- 완성된 v1 URL: 위 리소스 URL 뒤에 `/openai/v1/`을 붙인 형식

Entra 토큰이 잘못된 호스트로 전송되지 않도록 공개 Azure의
`*.openai.azure.com`, `*.services.ai.azure.com` 호스트만 허용합니다. URL에
사용자 정보나 비밀번호를 포함할 수 없으며, 포트를 명시할 때는 `443`만
허용합니다.

프로젝트 엔드포인트만 알고 있다면 기존 호환 변수도 사용할 수 있습니다.

```dotenv
AZURE_AI_PROJECT_ENDPOINT=https://your-resource.services.ai.azure.com/api/projects/your-project
```

`AZURE_OPENAI_ENDPOINT`와 `AZURE_AI_PROJECT_ENDPOINT`가 모두 설정되면 `AZURE_OPENAI_ENDPOINT`를 사용합니다.

### 선택 설정

```dotenv
# 비워 두면 모델별 지원 effort 전체를 높은 순서로 실행
REASONING_EFFORTS=xhigh,high,medium,low,none

# 일반 모델은 max_tokens, reasoning 모델은 max_completion_tokens에 적용
MAX_OUTPUT_TOKENS=4096

REQUEST_TIMEOUT_SECONDS=120

# 재시도 대기시간이 레이턴시에 섞이지 않도록 기본값은 0
MAX_RETRIES=0

# 기본값. 조직의 인증 지침에서 다른 scope를 요구할 때만 변경
AZURE_OPENAI_TOKEN_SCOPE=https://ai.azure.com/.default
```

## 3. 스모크 테스트

먼저 각 배포와 effort 조합을 짧게 확인합니다.

```bash
python test_latency.py --smoke --output latency_smoke.csv
```

워밍업은 배포마다 한 번 실행되며 측정 결과에는 포함되지 않습니다. 워밍업에서 인증, 엔드포인트, 배포 이름 오류가 발견되면 측정을 시작하지 않고 종료 코드 `1`을 반환합니다.

## 4. 전체 측정

각 조합과 프롬프트를 한 번씩 실행:

```bash
python test_latency.py
```

반복 횟수를 늘려 평균을 비교:

```bash
python test_latency.py --iterations 3
```

모델 간 수치를 비교할 때는 최소 3회 이상 반복을 권장합니다. 기본 1회 실행은 연결과 동작 확인용이며, 실행 시점의 일시적인 서비스 부하가 순위에 영향을 줄 수 있습니다.

워밍업을 생략하려면 `--skip-warmup`을 추가합니다.

## 결과 해석

- 레이턴시는 요청 전송 직전부터 비스트리밍 전체 응답 수신까지의 시간입니다. 첫 토큰 지연시간(TTFT)이 아닙니다.
- 콘솔에는 모델/effort별 평균, 최소, 최대, 표준편차, 평균 토큰, 평균 reasoning 토큰을 출력합니다.
- CSV에는 모델 ID와 실제 배포 이름을 별도 컬럼으로 저장하며, `response`는 공백을 정리한 앞 200자의 미리보기입니다.
- 개별 API 호출 실패와 `finish_reason=length` 같은 잘린 응답도 CSV에 기록하며, 하나라도 있으면 프로세스가 종료 코드 `1`을 반환합니다.
- reasoning 모델에는 `max_completion_tokens`, GPT-4.x 모델에는 `max_tokens`를 사용합니다.
- reasoning 모델에 지원되지 않는 `temperature`, `top_p`, `logprobs` 등의 파라미터는 전송하지 않습니다.

기본 출력 파일은 `latency_results.csv`이며 다음 컬럼을 포함합니다.

```text
model,deployment,prompt,reasoning_effort,latency_ms,tokens,completion_tokens,
prompt_tokens,reasoning_tokens,response,finish_reason,success,error,iteration,timestamp
```

## 문제 해결

| 증상 | 확인 사항 |
| --- | --- |
| `401` 또는 `403` | `az login` 상태, 구독/테넌트, `Cognitive Services OpenAI User` 역할, 토큰 scope |
| `404` 또는 deployment not found | `.env` 값이 모델 ID가 아니라 실제 Azure **배포 이름**인지 확인 |
| endpoint 형식 오류 | 리소스 엔드포인트 또는 `/api/projects/<project>` 형식을 사용했는지 확인 |
| 사용자 지정 프록시 또는 다른 클라우드 도메인 | 이 도구는 공개 Azure의 공식 OpenAI/Foundry 호스트만 허용 |
| effort 설정 오류 | `minimal`을 제거하고 지원 표의 값을 사용 |
| 일부 지역에서 모델 배포 불가 | Azure 모델 지역 가용성과 구독 quota 확인 |
| `finish_reason=length` 또는 빈 응답 | `MAX_OUTPUT_TOKENS`를 늘리거나 프롬프트의 응답 길이를 축소 |
| `finish_reason=content_filter` | 프롬프트와 Azure OpenAI 콘텐츠 필터 정책 확인 |
| GPT-5.6 `max` 또는 pro mode 측정 필요 | Responses API 전용 기능이므로 이 Chat Completions 실습과 별도로 측정 |

## 참고 문서

- [Azure OpenAI v1 API](https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle)
- [Azure OpenAI reasoning models](https://learn.microsoft.com/azure/foundry/openai/how-to/reasoning)
- [Azure OpenAI RBAC](https://learn.microsoft.com/azure/ai-foundry/openai/how-to/role-based-access-control)
- [OpenAI GPT-5.4 model](https://developers.openai.com/api/docs/models/gpt-5.4)
- [OpenAI GPT-5.6 model guidance](https://developers.openai.com/api/docs/guides/latest-model)
