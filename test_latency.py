"""Compare Azure OpenAI latency across configured model deployments."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlsplit

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import dotenv_values
from openai import OpenAI, OpenAIError


ApiName = Literal["chat_completions", "responses"]

CHAT_COMPLETIONS_API: ApiName = "chat_completions"
RESPONSES_API: ApiName = "responses"
EFFORT_ORDER = ("max", "xhigh", "high", "medium", "low", "none")
DEFAULT_TOKEN_SCOPE = "https://ai.azure.com/.default"
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 0
AZURE_OPENAI_HOST_SUFFIXES = (
    ".openai.azure.com",
    ".services.ai.azure.com",
)
AZURE_RESOURCE_NAME_PATTERN = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)

CSV_FIELDS = (
    "model",
    "deployment",
    "api",
    "prompt",
    "reasoning_effort",
    "latency_ms",
    "tokens",
    "output_tokens",
    "input_tokens",
    "reasoning_tokens",
    "response",
    "status",
    "incomplete_reason",
    "success",
    "error",
    "iteration",
    "timestamp",
)


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    api: ApiName
    supported_efforts: tuple[str, ...] = ()


@dataclass(frozen=True)
class TestConfiguration:
    model_id: str
    deployment_name: str
    api: ApiName
    reasoning_effort: str | None = None

    @property
    def label(self) -> str:
        model_label = self.model_id
        if self.deployment_name != self.model_id:
            model_label = f"{self.model_id} [{self.deployment_name}]"
        if self.reasoning_effort is not None:
            return f"{model_label} ({self.reasoning_effort})"
        return model_label


@dataclass(frozen=True)
class ParsedModelResponse:
    total_tokens: int | None
    output_tokens: int | None
    input_tokens: int | None
    reasoning_tokens: int | None
    content: str
    status: str | None
    incomplete_reason: str | None
    error: str | None


# Deployment names in Azure are user-defined, so capabilities must come from the
# environment-variable key rather than from the deployment-name value.
MODEL_DEFINITIONS: dict[str, ModelDefinition] = {
    "GPT_41_DEPLOYMENT_NAME": ModelDefinition("gpt-4.1", CHAT_COMPLETIONS_API),
    "GPT_4O_DEPLOYMENT_NAME": ModelDefinition("gpt-4o", CHAT_COMPLETIONS_API),
    "GPT_51_DEPLOYMENT_NAME": ModelDefinition(
        "gpt-5.1",
        RESPONSES_API,
        ("high", "medium", "low", "none"),
    ),
    "GPT_52_DEPLOYMENT_NAME": ModelDefinition(
        "gpt-5.2",
        RESPONSES_API,
        ("xhigh", "high", "medium", "low", "none"),
    ),
    "GPT_54_DEPLOYMENT_NAME": ModelDefinition(
        "gpt-5.4",
        RESPONSES_API,
        ("xhigh", "high", "medium", "low", "none"),
    ),
    "GPT_54_MINI_DEPLOYMENT_NAME": ModelDefinition(
        "gpt-5.4-mini",
        RESPONSES_API,
        ("xhigh", "high", "medium", "low", "none"),
    ),
    "GPT_54_NANO_DEPLOYMENT_NAME": ModelDefinition(
        "gpt-5.4-nano",
        RESPONSES_API,
        ("xhigh", "high", "medium", "low", "none"),
    ),
    "GPT_56_SOL_DEPLOYMENT_NAME": ModelDefinition(
        "gpt-5.6-sol",
        RESPONSES_API,
        ("max", "xhigh", "high", "medium", "low", "none"),
    ),
    "GPT_56_TERRA_DEPLOYMENT_NAME": ModelDefinition(
        "gpt-5.6-terra",
        RESPONSES_API,
        ("max", "xhigh", "high", "medium", "low", "none"),
    ),
    "GPT_56_LUNA_DEPLOYMENT_NAME": ModelDefinition(
        "gpt-5.6-luna",
        RESPONSES_API,
        ("max", "xhigh", "high", "medium", "low", "none"),
    ),
}

TEST_PROMPTS = (
    "프랑스의 수도는 어디인가요?",
    "양자 컴퓨팅을 비전공자에게 5문장 이내로 설명해주세요.",
    "팩토리얼을 계산하는 파이썬 함수와 핵심 설명을 간결하게 작성해주세요.",
)


def normalize_azure_openai_base_url(endpoint: str) -> str:
    """Convert a resource or Foundry project endpoint to an Azure OpenAI v1 URL."""
    value = endpoint.strip()
    if not value:
        raise ValueError("Azure OpenAI endpoint is empty.")

    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Azure OpenAI endpoint must be an absolute HTTPS URL.")
    if parsed.query or parsed.fragment:
        raise ValueError("Azure OpenAI endpoint must not contain a query or fragment.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Azure OpenAI endpoint must not contain credentials.")

    hostname = parsed.hostname
    host_suffix = next(
        (
            suffix
            for suffix in AZURE_OPENAI_HOST_SUFFIXES
            if hostname is not None and hostname.endswith(suffix)
        ),
        None,
    )
    if hostname is None or host_suffix is None:
        allowed_hosts = ", ".join(f"*{suffix}" for suffix in AZURE_OPENAI_HOST_SUFFIXES)
        raise ValueError(
            f"Azure OpenAI endpoint host must match one of: {allowed_hosts}."
        )
    resource_name = hostname[: -len(host_suffix)]
    if AZURE_RESOURCE_NAME_PATTERN.fullmatch(resource_name) is None:
        raise ValueError(
            "Azure OpenAI endpoint must contain a valid Azure resource name."
        )
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Azure OpenAI endpoint contains an invalid port.") from error
    if port not in (None, 443):
        raise ValueError("Azure OpenAI endpoint may only use HTTPS port 443.")

    path = parsed.path.rstrip("/")
    if path.startswith("/api/projects"):
        parts = path.split("/")
        if (
            host_suffix != ".services.ai.azure.com"
            or len(parts) != 4
            or not parts[3]
        ):
            raise ValueError(
                "Foundry project endpoint must use "
                "https://<resource>.services.ai.azure.com/api/projects/<project>."
            )
        path = ""
    elif path not in ("", "/openai/v1"):
        raise ValueError(
            "Azure OpenAI endpoint must be a resource endpoint, a Foundry project "
            "endpoint, or a URL ending in /openai/v1/."
        )

    return f"{parsed.scheme}://{hostname}/openai/v1/"


def get_endpoint(env: Mapping[str, str]) -> tuple[str, str]:
    """Return the configured endpoint and the environment variable that supplied it."""
    for key in ("AZURE_OPENAI_ENDPOINT", "AZURE_AI_PROJECT_ENDPOINT"):
        value = (env.get(key) or "").strip()
        if value:
            return value, key
    raise ValueError(
        "Set AZURE_OPENAI_ENDPOINT or AZURE_AI_PROJECT_ENDPOINT in your .env file."
    )


def get_reasoning_efforts(env: Mapping[str, str] | None = None) -> list[str]:
    """Return requested effort levels in deterministic high-to-low order."""
    source = os.environ if env is None else env
    raw = (source.get("REASONING_EFFORTS") or "").strip()
    if not raw:
        return list(EFFORT_ORDER)

    requested = {value.strip().lower() for value in raw.split(",") if value.strip()}
    if not requested:
        raise ValueError("REASONING_EFFORTS must contain at least one value.")

    unsupported = sorted(requested.difference(EFFORT_ORDER))
    if unsupported:
        valid = ", ".join(EFFORT_ORDER)
        raise ValueError(
            f"Unsupported REASONING_EFFORTS value(s): {', '.join(unsupported)}. "
            f"Valid values for this guide are: {valid}."
        )

    return [effort for effort in EFFORT_ORDER if effort in requested]


def normalize_model_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Normalize model variable names case-insensitively and reject duplicates."""
    normalized: dict[str, str] = {}
    original_keys: dict[str, str] = {}
    for key, value in env.items():
        normalized_key = key.upper()
        if not (
            normalized_key.startswith("GPT_")
            and normalized_key.endswith("_DEPLOYMENT_NAME")
        ):
            continue

        normalized_value = value.strip()
        previous_value = normalized.get(normalized_key)
        if (
            previous_value is not None
            and previous_value
            and normalized_value
            and previous_value != normalized_value
        ):
            raise ValueError(
                f"Conflicting values for {original_keys[normalized_key]} and {key}."
            )
        if previous_value is None or normalized_value:
            normalized[normalized_key] = normalized_value
            original_keys[normalized_key] = key
    return normalized


def load_configuration_environment(
    dotenv_path: str | Path = ".env",
    process_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Load .env settings, then apply process-environment overrides."""
    file_environment = {
        key: value
        for key, value in dotenv_values(dotenv_path).items()
        if value is not None
    }
    runtime_environment = dict(os.environ if process_env is None else process_env)

    file_models = normalize_model_environment(file_environment)
    runtime_models = normalize_model_environment(runtime_environment)

    combined = dict(file_environment)
    combined.update(runtime_environment)
    for key in list(combined):
        normalized_key = key.upper()
        if (
            normalized_key.startswith("GPT_")
            and normalized_key.endswith("_DEPLOYMENT_NAME")
        ):
            del combined[key]
    combined.update(file_models)
    combined.update(runtime_models)
    return combined


def build_model_configs(
    env: Mapping[str, str] | None = None,
    efforts: Sequence[str] | None = None,
) -> list[TestConfiguration]:
    """Expand configured deployments into valid model and effort combinations."""
    source = os.environ if env is None else env
    model_environment = normalize_model_environment(source)
    requested_efforts = (
        list(efforts) if efforts is not None else get_reasoning_efforts(source)
    )

    unknown_keys = sorted(
        key
        for key, value in model_environment.items()
        if (value or "").strip() and key not in MODEL_DEFINITIONS
    )
    if unknown_keys:
        raise ValueError(
            "Unsupported deployment environment variable(s): "
            f"{', '.join(unknown_keys)}. Add the model to MODEL_DEFINITIONS before use."
        )

    configurations: list[TestConfiguration] = []
    deployment_owners: dict[str, str] = {}

    for env_key, definition in MODEL_DEFINITIONS.items():
        deployment_name = (model_environment.get(env_key) or "").strip()
        if not deployment_name:
            continue

        previous_model = deployment_owners.get(deployment_name)
        if previous_model is not None:
            raise ValueError(
                f"Deployment '{deployment_name}' is configured for both "
                f"{previous_model} and {definition.model_id}."
            )
        deployment_owners[deployment_name] = definition.model_id

        if not definition.supported_efforts:
            configurations.append(
                TestConfiguration(
                    definition.model_id,
                    deployment_name,
                    definition.api,
                )
            )
            continue

        model_efforts = [
            effort
            for effort in requested_efforts
            if effort in definition.supported_efforts
        ]
        if not model_efforts:
            supported = ", ".join(definition.supported_efforts)
            raise ValueError(
                f"No requested reasoning effort is supported by {definition.model_id}. "
                f"Supported values: {supported}."
            )

        configurations.extend(
            TestConfiguration(
                definition.model_id,
                deployment_name,
                definition.api,
                effort,
            )
            for effort in model_efforts
        )

    if not configurations:
        expected = ", ".join(MODEL_DEFINITIONS)
        raise ValueError(
            "No model deployments found. Set at least one supported deployment "
            f"variable in .env: {expected}."
        )

    return configurations


def get_positive_int(
    env: Mapping[str, str], key: str, default: int, *, minimum: int = 1
) -> int:
    raw = (env.get(key) or str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer.") from error
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}.")
    return value


def get_positive_float(
    env: Mapping[str, str], key: str, default: float
) -> float:
    raw = (env.get(key) or str(default)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{key} must be a number.") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a finite number greater than 0.")
    return value


def build_chat_completion_params(
    configuration: TestConfiguration, prompt: str, max_output_tokens: int
) -> dict[str, Any]:
    """Build a GPT-4.x Chat Completions request."""
    if configuration.api != CHAT_COMPLETIONS_API:
        raise ValueError("Chat Completions parameters require a GPT-4.x configuration.")
    if configuration.reasoning_effort is not None:
        raise ValueError("GPT-4.x Chat Completions must not set reasoning_effort.")
    return {
        "model": configuration.deployment_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
    }


def build_responses_params(
    configuration: TestConfiguration, prompt: str, max_output_tokens: int
) -> dict[str, Any]:
    """Build a GPT-5.x Responses API request."""
    if configuration.api != RESPONSES_API:
        raise ValueError("Responses parameters require a GPT-5.x configuration.")
    if configuration.reasoning_effort is None:
        raise ValueError("GPT-5.x Responses requests require reasoning_effort.")
    return {
        "model": configuration.deployment_name,
        "input": prompt,
        "reasoning": {"effort": configuration.reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "store": False,
    }


def get_chat_completion_error(
    finish_reason: str | None, content: str
) -> str | None:
    """Return an actionable error for an incomplete Chat Completions response."""
    if finish_reason == "length":
        return (
            "Incomplete response (finish_reason='length'). "
            "Increase MAX_OUTPUT_TOKENS or shorten the prompt."
        )
    if finish_reason == "content_filter":
        return "The response was blocked by the Azure OpenAI content filter."
    if finish_reason != "stop":
        return f"Incomplete response (finish_reason={finish_reason!r})."
    if not content.strip():
        return "The model returned an empty response."
    return None


def get_responses_error(
    status: str | None,
    incomplete_reason: str | None,
    content: str,
    failure: str | None = None,
) -> str | None:
    """Return an actionable error for an incomplete Responses API response."""
    if status == "completed":
        if not content.strip():
            return "The model returned an empty response."
        return None
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return (
                "Incomplete response (reason='max_output_tokens'). "
                "Increase MAX_OUTPUT_TOKENS or shorten the prompt."
            )
        if incomplete_reason == "content_filter":
            return "The response was blocked by the Azure OpenAI content filter."
        return f"Incomplete response (reason={incomplete_reason!r})."
    if status == "failed" and failure:
        return f"Response failed: {failure}"
    if status == "failed":
        return "Response failed without error details."
    return f"Incomplete response (status={status!r})."


def parse_chat_completion_response(response: Any) -> ParsedModelResponse:
    """Normalize a Chat Completions response."""
    usage = response.usage
    reasoning_tokens = None
    if usage is not None:
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

    choice = response.choices[0] if response.choices else None
    content = ""
    finish_reason = None
    if choice is not None:
        content = choice.message.content or ""
        finish_reason = choice.finish_reason

    return ParsedModelResponse(
        total_tokens=usage.total_tokens if usage is not None else None,
        output_tokens=usage.completion_tokens if usage is not None else None,
        input_tokens=usage.prompt_tokens if usage is not None else None,
        reasoning_tokens=reasoning_tokens,
        content=content,
        status=finish_reason,
        incomplete_reason=None,
        error=get_chat_completion_error(finish_reason, content),
    )


def parse_responses_api_response(response: Any) -> ParsedModelResponse:
    """Normalize a Responses API response."""
    usage = response.usage
    reasoning_tokens = None
    if usage is not None:
        details = getattr(usage, "output_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

    incomplete_details = response.incomplete_details
    incomplete_reason = (
        incomplete_details.reason if incomplete_details is not None else None
    )
    response_error = response.error
    failure = None
    if response_error is not None:
        code = getattr(response_error, "code", None)
        message = getattr(response_error, "message", None)
        if code and message:
            failure = f"{code}: {message}"
        elif message:
            failure = str(message)
        elif code:
            failure = str(code)

    content = response.output_text or ""
    status = response.status
    return ParsedModelResponse(
        total_tokens=usage.total_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
        input_tokens=usage.input_tokens if usage is not None else None,
        reasoning_tokens=reasoning_tokens,
        content=content,
        status=status,
        incomplete_reason=incomplete_reason,
        error=get_responses_error(status, incomplete_reason, content, failure),
    )


class LatencyTester:
    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        *,
        prompts: Sequence[str] = TEST_PROMPTS,
    ) -> None:
        self.env = dict(os.environ if env is None else env)
        endpoint, endpoint_source = get_endpoint(self.env)
        self.base_url = normalize_azure_openai_base_url(endpoint)
        self.reasoning_efforts = get_reasoning_efforts(self.env)
        self.configurations = build_model_configs(
            self.env, efforts=self.reasoning_efforts
        )
        self.prompts = tuple(prompts)
        if not self.prompts:
            raise ValueError("At least one test prompt is required.")

        self.max_output_tokens = get_positive_int(
            self.env, "MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS
        )
        timeout = get_positive_float(
            self.env,
            "REQUEST_TIMEOUT_SECONDS",
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        max_retries = get_positive_int(
            self.env, "MAX_RETRIES", DEFAULT_MAX_RETRIES, minimum=0
        )
        token_scope = (
            self.env.get("AZURE_OPENAI_TOKEN_SCOPE") or DEFAULT_TOKEN_SCOPE
        ).strip()
        if not token_scope.endswith("/.default"):
            raise ValueError("AZURE_OPENAI_TOKEN_SCOPE must end with '/.default'.")

        credential = DefaultAzureCredential()
        client_created = False
        try:
            token_provider = get_bearer_token_provider(credential, token_scope)
            client = OpenAI(
                base_url=self.base_url,
                api_key=token_provider,
                timeout=timeout,
                max_retries=max_retries,
            )
            client_created = True
        finally:
            if not client_created:
                credential.close()
        self.credential = credential
        self.client = client

        print("Azure OpenAI latency comparison")
        print(f"  Endpoint source: {endpoint_source}")
        print("  API:             /openai/v1/ (no dated api-version)")
        print("  Routing:         GPT-4.x Chat Completions; GPT-5.x Responses")
        print(f"  Token scope:     {token_scope}")
        print(f"  Output limit:    {self.max_output_tokens} tokens")
        print(f"  Configurations:  {len(self.configurations)}")
        for configuration in self.configurations:
            print(f"    - {configuration.label} via {configuration.api}")

    def close(self) -> None:
        try:
            self.client.close()
        finally:
            self.credential.close()

    def _create_model_response(
        self,
        configuration: TestConfiguration,
        prompt: str,
        max_output_tokens: int,
    ) -> Any:
        if configuration.api == CHAT_COMPLETIONS_API:
            return self.client.chat.completions.create(
                **build_chat_completion_params(
                    configuration,
                    prompt,
                    max_output_tokens,
                )
            )
        if configuration.api == RESPONSES_API:
            return self.client.responses.create(
                **build_responses_params(
                    configuration,
                    prompt,
                    max_output_tokens,
                )
            )
        raise ValueError(f"Unsupported API routing value: {configuration.api!r}.")

    def warmup_clients(self) -> None:
        """Warm each deployment once and fail before the measured run on an error."""
        print(f"\n{'=' * 80}")
        print("Warming up deployments")
        print(f"{'=' * 80}")

        by_deployment: dict[str, list[TestConfiguration]] = {}
        for configuration in self.configurations:
            by_deployment.setdefault(configuration.deployment_name, []).append(
                configuration
            )

        failures: list[str] = []
        for deployment_name, configurations in by_deployment.items():
            warmup_configuration = next(
                (
                    configuration
                    for configuration in configurations
                    if configuration.reasoning_effort == "none"
                ),
                configurations[-1],
            )
            print(f"Warming up {warmup_configuration.label}...", end=" ", flush=True)
            started = time.perf_counter()
            try:
                self._create_model_response(
                    warmup_configuration,
                    "Reply with OK.",
                    min(self.max_output_tokens, 32),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                print(f"{elapsed_ms:.0f}ms")
            except (OpenAIError, AzureError) as error:
                print("failed")
                failures.append(f"{deployment_name}: {error}")

        if failures:
            details = "\n".join(f"  - {failure}" for failure in failures)
            raise RuntimeError(f"Warmup failed:\n{details}")

    def test_model_latency(
        self, configuration: TestConfiguration, prompt: str
    ) -> dict[str, Any]:
        """Measure one non-streaming request from send to complete response."""
        started = time.perf_counter()
        try:
            response = self._create_model_response(
                configuration,
                prompt,
                self.max_output_tokens,
            )
            latency_ms = (time.perf_counter() - started) * 1000
        except (OpenAIError, AzureError) as error:
            return {
                "model": configuration.model_id,
                "deployment": configuration.deployment_name,
                "api": configuration.api,
                "prompt": prompt,
                "reasoning_effort": configuration.reasoning_effort,
                "latency_ms": None,
                "tokens": None,
                "output_tokens": None,
                "input_tokens": None,
                "reasoning_tokens": None,
                "response": None,
                "status": None,
                "incomplete_reason": None,
                "success": False,
                "error": str(error),
            }

        if configuration.api == CHAT_COMPLETIONS_API:
            parsed = parse_chat_completion_response(response)
        else:
            parsed = parse_responses_api_response(response)
        response_preview = " ".join(parsed.content.split())[:200]

        return {
            "model": configuration.model_id,
            "deployment": configuration.deployment_name,
            "api": configuration.api,
            "prompt": prompt,
            "reasoning_effort": configuration.reasoning_effort,
            "latency_ms": round(latency_ms, 2),
            "tokens": parsed.total_tokens,
            "output_tokens": parsed.output_tokens,
            "input_tokens": parsed.input_tokens,
            "reasoning_tokens": parsed.reasoning_tokens,
            "response": response_preview,
            "status": parsed.status,
            "incomplete_reason": parsed.incomplete_reason,
            "success": parsed.error is None,
            "error": parsed.error,
        }

    def run_tests(self, iterations: int = 1) -> list[dict[str, Any]]:
        """Run every configured model/effort and prompt combination."""
        results: list[dict[str, Any]] = []

        print(f"\n{'=' * 80}")
        print("Measured run")
        print(f"  Configurations: {len(self.configurations)}")
        print(f"  Prompts:        {len(self.prompts)}")
        print(f"  Iterations:     {iterations}")
        print(f"{'=' * 80}")

        for configuration in self.configurations:
            print(f"\nTesting {configuration.label}")
            for prompt_index, prompt in enumerate(self.prompts, 1):
                print(f"  Prompt {prompt_index}/{len(self.prompts)}: {prompt}")
                for iteration in range(1, iterations + 1):
                    result = self.test_model_latency(configuration, prompt)
                    result["iteration"] = iteration
                    result["timestamp"] = datetime.now(timezone.utc).isoformat()
                    results.append(result)

                    if result["success"]:
                        reasoning = ""
                        if result["reasoning_tokens"] is not None:
                            reasoning = (
                                f", reasoning={result['reasoning_tokens']}"
                            )
                        print(
                            f"    iteration {iteration}: "
                            f"{result['latency_ms']:.0f}ms, "
                            f"tokens={result['tokens']}{reasoning}"
                        )
                    else:
                        print(
                            f"    iteration {iteration}: failed: {result['error']}"
                        )

                    time.sleep(0.2)

        return results

    def analyze_results(self, results: Sequence[Mapping[str, Any]]) -> None:
        """Print latency and token summaries for successful requests."""
        print(f"\n{'=' * 80}")
        print("Results")
        print(f"{'=' * 80}")

        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for result in results:
            if result["success"]:
                effort = result.get("reasoning_effort")
                label = str(result["model"])
                if effort is not None:
                    label = f"{label} ({effort})"
                grouped.setdefault(label, []).append(result)

        if grouped:
            print(
                f"{'Model':<28} {'Avg ms':>10} {'Min ms':>10} {'Max ms':>10} "
                f"{'Std dev':>10} {'Avg tokens':>12} {'Avg reason':>12} {'Tests':>7}"
            )
            print("-" * 115)
            rows: list[tuple[float, str, list[Mapping[str, Any]]]] = []
            for label, model_results in grouped.items():
                latencies = [float(result["latency_ms"]) for result in model_results]
                rows.append((fmean(latencies), label, model_results))

            for average_latency, label, model_results in sorted(rows):
                latencies = [float(result["latency_ms"]) for result in model_results]
                token_values = [
                    int(result["tokens"])
                    for result in model_results
                    if result["tokens"] is not None
                ]
                reasoning_values = [
                    int(result["reasoning_tokens"])
                    for result in model_results
                    if result["reasoning_tokens"] is not None
                ]
                average_tokens = (
                    f"{fmean(token_values):.1f}" if token_values else "-"
                )
                average_reasoning = (
                    f"{fmean(reasoning_values):.1f}" if reasoning_values else "-"
                )
                print(
                    f"{label:<28} {average_latency:>10.2f} "
                    f"{min(latencies):>10.2f} {max(latencies):>10.2f} "
                    f"{pstdev(latencies):>10.2f} {average_tokens:>12} "
                    f"{average_reasoning:>12} {len(latencies):>7}"
                )
        else:
            print("No successful requests.")

        errors = [result for result in results if not result["success"]]
        if errors:
            print(f"\nERRORS ({len(errors)})")
            seen: set[tuple[Any, Any, Any]] = set()
            for error in errors:
                key = (
                    error["model"],
                    error.get("reasoning_effort"),
                    error["error"],
                )
                if key in seen:
                    continue
                seen.add(key)
                effort = error.get("reasoning_effort")
                label = str(error["model"])
                if effort is not None:
                    label = f"{label} ({effort})"
                print(f"  {label}: {error['error']}")

    @staticmethod
    def save_results(
        results: Sequence[Mapping[str, Any]],
        filename: str | Path = "latency_results.csv",
    ) -> None:
        """Save results with a stable CSV schema."""
        if not results:
            raise ValueError("No results to save.")

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_path}")


def positive_int_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Azure OpenAI deployment latency by reasoning effort."
    )
    parser.add_argument(
        "--iterations",
        type=positive_int_argument,
        default=1,
        help="Measured requests per prompt and configuration (default: 1).",
    )
    parser.add_argument(
        "--output",
        default="latency_results.csv",
        help="CSV output path (default: latency_results.csv).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use only the first prompt for a shorter end-to-end check.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the unmeasured deployment warmup.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prompts = TEST_PROMPTS[:1] if args.smoke else TEST_PROMPTS
    tester: LatencyTester | None = None

    try:
        tester = LatencyTester(
            env=load_configuration_environment(),
            prompts=prompts,
        )
        if not args.skip_warmup:
            tester.warmup_clients()
        results = tester.run_tests(iterations=args.iterations)
        tester.analyze_results(results)
        tester.save_results(results, args.output)
        return 1 if any(not result["success"] for result in results) else 0
    except (ValueError, RuntimeError, OpenAIError, AzureError) as error:
        print(f"Error: {error}")
        return 1
    finally:
        if tester is not None:
            tester.close()


if __name__ == "__main__":
    raise SystemExit(main())
