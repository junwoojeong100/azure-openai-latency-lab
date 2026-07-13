import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from test_latency import (
    CHAT_COMPLETIONS_API,
    CSV_FIELDS,
    RESPONSES_API,
    TestConfiguration,
    build_chat_completion_params,
    build_model_configs,
    build_responses_params,
    get_chat_completion_error,
    get_positive_float,
    get_reasoning_efforts,
    get_responses_error,
    load_configuration_environment,
    normalize_azure_openai_base_url,
    parse_chat_completion_response,
    parse_responses_api_response,
    LatencyTester,
)


class EndpointNormalizationTests(unittest.TestCase):
    def test_resource_endpoint(self):
        self.assertEqual(
            normalize_azure_openai_base_url(
                "https://example.openai.azure.com/"
            ),
            "https://example.openai.azure.com/openai/v1/",
        )

    def test_foundry_project_endpoint(self):
        self.assertEqual(
            normalize_azure_openai_base_url(
                "https://example.services.ai.azure.com/api/projects/demo"
            ),
            "https://example.services.ai.azure.com/openai/v1/",
        )

    def test_foundry_resource_endpoint(self):
        self.assertEqual(
            normalize_azure_openai_base_url(
                "https://example.services.ai.azure.com/"
            ),
            "https://example.services.ai.azure.com/openai/v1/",
        )

    def test_complete_v1_endpoint(self):
        self.assertEqual(
            normalize_azure_openai_base_url(
                "https://example.services.ai.azure.com/openai/v1/"
            ),
            "https://example.services.ai.azure.com/openai/v1/",
        )

    def test_explicit_https_port_is_normalized(self):
        self.assertEqual(
            normalize_azure_openai_base_url(
                "https://example.openai.azure.com:443/"
            ),
            "https://example.openai.azure.com/openai/v1/",
        )

    def test_rejects_dated_api_path(self):
        with self.assertRaisesRegex(ValueError, "resource endpoint"):
            normalize_azure_openai_base_url(
                "https://example.openai.azure.com/openai/deployments/demo"
            )

    def test_rejects_non_azure_host(self):
        with self.assertRaisesRegex(ValueError, "host must match"):
            normalize_azure_openai_base_url("https://example.com")

    def test_rejects_malformed_resource_name(self):
        for hostname in (
            " example.openai.azure.com",
            "example .openai.azure.com",
            "_example.openai.azure.com",
            "-example.openai.azure.com",
            "example-.openai.azure.com",
            "example..openai.azure.com",
        ):
            with self.subTest(hostname=hostname):
                with self.assertRaisesRegex(ValueError, "resource name"):
                    normalize_azure_openai_base_url(f"https://{hostname}")

    def test_rejects_embedded_credentials(self):
        with self.assertRaisesRegex(ValueError, "must not contain credentials"):
            normalize_azure_openai_base_url(
                "https://user:password@example.openai.azure.com"
            )

    def test_rejects_non_https_port(self):
        with self.assertRaisesRegex(ValueError, "port 443"):
            normalize_azure_openai_base_url(
                "https://example.openai.azure.com:8443"
            )

    def test_rejects_invalid_port(self):
        with self.assertRaisesRegex(ValueError, "invalid port"):
            normalize_azure_openai_base_url(
                "https://example.openai.azure.com:not-a-port"
            )

    def test_rejects_malformed_foundry_project_path(self):
        with self.assertRaisesRegex(ValueError, "Foundry project endpoint"):
            normalize_azure_openai_base_url(
                "https://example.services.ai.azure.com/api/projects/demo/extra"
            )

    def test_rejects_project_path_on_openai_resource_host(self):
        with self.assertRaisesRegex(ValueError, "Foundry project endpoint"):
            normalize_azure_openai_base_url(
                "https://example.openai.azure.com/api/projects/demo"
            )


class ModelConfigurationTests(unittest.TestCase):
    def test_default_efforts_use_current_high_to_low_order(self):
        self.assertEqual(
            get_reasoning_efforts({}),
            ["max", "xhigh", "high", "medium", "low", "none"],
        )

    def test_minimal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "minimal"):
            get_reasoning_efforts({"REASONING_EFFORTS": "high,minimal"})

    def test_max_effort_is_accepted(self):
        self.assertEqual(
            get_reasoning_efforts({"REASONING_EFFORTS": "max"}),
            ["max"],
        )

    def test_capabilities_do_not_depend_on_deployment_name(self):
        configurations = build_model_configs(
            {
                "GPT_51_DEPLOYMENT_NAME": "arbitrary-prod-a",
                "GPT_54_DEPLOYMENT_NAME": "arbitrary-prod-b",
                "GPT_56_SOL_DEPLOYMENT_NAME": "arbitrary-prod-c",
            },
            efforts=["xhigh", "none"],
        )
        self.assertEqual(
            configurations,
            [
                TestConfiguration(
                    "gpt-5.1",
                    "arbitrary-prod-a",
                    RESPONSES_API,
                    "none",
                ),
                TestConfiguration(
                    "gpt-5.4",
                    "arbitrary-prod-b",
                    RESPONSES_API,
                    "xhigh",
                ),
                TestConfiguration(
                    "gpt-5.4",
                    "arbitrary-prod-b",
                    RESPONSES_API,
                    "none",
                ),
                TestConfiguration(
                    "gpt-5.6-sol",
                    "arbitrary-prod-c",
                    RESPONSES_API,
                    "xhigh",
                ),
                TestConfiguration(
                    "gpt-5.6-sol",
                    "arbitrary-prod-c",
                    RESPONSES_API,
                    "none",
                ),
            ],
        )

    def test_all_gpt_56_models_support_max_effort(self):
        cases = {
            "GPT_56_SOL_DEPLOYMENT_NAME": "gpt-5.6-sol",
            "GPT_56_TERRA_DEPLOYMENT_NAME": "gpt-5.6-terra",
            "GPT_56_LUNA_DEPLOYMENT_NAME": "gpt-5.6-luna",
        }
        for env_key, model_id in cases.items():
            with self.subTest(model=model_id):
                self.assertEqual(
                    build_model_configs(
                        {env_key: f"{model_id}-deployment"},
                        efforts=["max"],
                    ),
                    [
                        TestConfiguration(
                            model_id,
                            f"{model_id}-deployment",
                            RESPONSES_API,
                            "max",
                        )
                    ],
                )

    def test_gpt_56_responses_effort_matrix(self):
        configurations = build_model_configs(
            {"GPT_56_TERRA_DEPLOYMENT_NAME": "terra-deployment"}
        )
        self.assertEqual(
            [item.reasoning_effort for item in configurations],
            ["max", "xhigh", "high", "medium", "low", "none"],
        )
        self.assertTrue(
            all(item.api == RESPONSES_API for item in configurations)
        )

    def test_gpt_4_and_gpt_5_use_different_apis(self):
        configurations = build_model_configs(
            {
                "GPT_41_DEPLOYMENT_NAME": "deployment-41",
                "GPT_51_DEPLOYMENT_NAME": "deployment-51",
            },
            efforts=["none"],
        )
        self.assertEqual(
            configurations,
            [
                TestConfiguration(
                    "gpt-4.1",
                    "deployment-41",
                    CHAT_COMPLETIONS_API,
                ),
                TestConfiguration(
                    "gpt-5.1",
                    "deployment-51",
                    RESPONSES_API,
                    "none",
                ),
            ],
        )

    def test_legacy_mixed_case_gpt_4o_key_is_supported(self):
        self.assertEqual(
            build_model_configs({"GPT_4o_DEPLOYMENT_NAME": "deployment-4o"}),
            [
                TestConfiguration(
                    "gpt-4o",
                    "deployment-4o",
                    CHAT_COMPLETIONS_API,
                )
            ],
        )

    def test_conflicting_gpt_4o_key_spellings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting values"):
            build_model_configs(
                {
                    "GPT_4O_DEPLOYMENT_NAME": "deployment-a",
                    "GPT_4o_DEPLOYMENT_NAME": "deployment-b",
                }
            )

    def test_equivalent_alias_values_ignore_surrounding_whitespace(self):
        self.assertEqual(
            build_model_configs(
                {
                    "GPT_4O_DEPLOYMENT_NAME": " deployment-4o ",
                    "GPT_4o_DEPLOYMENT_NAME": "deployment-4o",
                }
            ),
            [
                TestConfiguration(
                    "gpt-4o",
                    "deployment-4o",
                    CHAT_COMPLETIONS_API,
                )
            ],
        )

    def test_unknown_model_variable_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "GPT_55_DEPLOYMENT_NAME"):
            build_model_configs(
                {"GPT_55_DEPLOYMENT_NAME": "gpt-5.5"},
                efforts=["high"],
            )

    def test_duplicate_deployment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "configured for both"):
            build_model_configs(
                {
                    "GPT_41_DEPLOYMENT_NAME": "shared",
                    "GPT_4o_DEPLOYMENT_NAME": "shared",
                }
            )


class EnvironmentLoadingTests(unittest.TestCase):
    def test_process_environment_overrides_dotenv_case_insensitively(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text(
                "GPT_4o_DEPLOYMENT_NAME=file-deployment\n",
                encoding="utf-8",
            )
            environment = load_configuration_environment(
                dotenv_path,
                {"GPT_4O_DEPLOYMENT_NAME": "runtime-deployment"},
            )
            self.assertEqual(
                environment["GPT_4O_DEPLOYMENT_NAME"],
                "runtime-deployment",
            )
            self.assertNotIn("GPT_4o_DEPLOYMENT_NAME", environment)

    def test_dotenv_case_conflict_is_rejected_before_os_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text(
                "GPT_4O_DEPLOYMENT_NAME=deployment-a\n"
                "GPT_4o_DEPLOYMENT_NAME=deployment-b\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Conflicting values"):
                load_configuration_environment(dotenv_path, {})


class RequestParameterTests(unittest.TestCase):
    def test_gpt_4_uses_chat_completions_parameters(self):
        params = build_chat_completion_params(
            TestConfiguration(
                "gpt-4.1",
                "deployment-41",
                CHAT_COMPLETIONS_API,
            ),
            "hello",
            128,
        )
        self.assertEqual(params["max_tokens"], 128)
        self.assertEqual(
            params["messages"],
            [{"role": "user", "content": "hello"}],
        )
        self.assertNotIn("max_output_tokens", params)
        self.assertNotIn("reasoning", params)

    def test_gpt_5_uses_responses_parameters(self):
        params = build_responses_params(
            TestConfiguration(
                "gpt-5.6-sol",
                "deployment-56-sol",
                RESPONSES_API,
                "max",
            ),
            "hello",
            128,
        )
        self.assertEqual(params["max_output_tokens"], 128)
        self.assertEqual(params["input"], "hello")
        self.assertEqual(params["reasoning"], {"effort": "max"})
        self.assertFalse(params["store"])
        self.assertNotIn("max_tokens", params)
        self.assertNotIn("messages", params)


class NumericConfigurationTests(unittest.TestCase):
    def test_positive_finite_float_is_accepted(self):
        self.assertEqual(
            get_positive_float({"TIMEOUT": "1.5"}, "TIMEOUT", 10.0),
            1.5,
        )

    def test_non_finite_float_is_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite number"):
                    get_positive_float({"TIMEOUT": value}, "TIMEOUT", 10.0)


class ResponseValidationTests(unittest.TestCase):
    def test_complete_chat_text_is_accepted(self):
        self.assertIsNone(get_chat_completion_error("stop", "완료"))

    def test_chat_length_truncation_is_rejected(self):
        self.assertIn(
            "MAX_OUTPUT_TOKENS",
            get_chat_completion_error("length", "") or "",
        )

    def test_empty_chat_text_is_rejected(self):
        self.assertEqual(
            get_chat_completion_error("stop", "  "),
            "The model returned an empty response.",
        )

    def test_chat_content_filter_has_specific_error(self):
        self.assertIn(
            "content filter",
            (get_chat_completion_error("content_filter", "") or "").lower(),
        )

    def test_complete_responses_text_is_accepted(self):
        self.assertIsNone(
            get_responses_error("completed", None, "완료")
        )

    def test_responses_output_limit_is_rejected(self):
        self.assertIn(
            "MAX_OUTPUT_TOKENS",
            get_responses_error(
                "incomplete",
                "max_output_tokens",
                "",
            )
            or "",
        )

    def test_responses_content_filter_has_specific_error(self):
        self.assertIn(
            "content filter",
            (
                get_responses_error(
                    "incomplete",
                    "content_filter",
                    "",
                )
                or ""
            ).lower(),
        )

    def test_failed_response_surfaces_service_error(self):
        self.assertEqual(
            get_responses_error(
                "failed",
                None,
                "",
                "server_error: unavailable",
            ),
            "Response failed: server_error: unavailable",
        )


class ResponseParsingTests(unittest.TestCase):
    def test_chat_completion_response_is_normalized(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                total_tokens=12,
                completion_tokens=7,
                prompt_tokens=5,
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=2
                ),
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="완료"),
                )
            ],
        )

        parsed = parse_chat_completion_response(response)

        self.assertEqual(parsed.total_tokens, 12)
        self.assertEqual(parsed.output_tokens, 7)
        self.assertEqual(parsed.input_tokens, 5)
        self.assertEqual(parsed.reasoning_tokens, 2)
        self.assertEqual(parsed.status, "stop")
        self.assertIsNone(parsed.error)

    def test_responses_api_response_is_normalized(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                total_tokens=18,
                output_tokens=11,
                input_tokens=7,
                output_tokens_details=SimpleNamespace(
                    reasoning_tokens=4
                ),
            ),
            output_text="완료",
            status="completed",
            incomplete_details=None,
            error=None,
        )

        parsed = parse_responses_api_response(response)

        self.assertEqual(parsed.total_tokens, 18)
        self.assertEqual(parsed.output_tokens, 11)
        self.assertEqual(parsed.input_tokens, 7)
        self.assertEqual(parsed.reasoning_tokens, 4)
        self.assertEqual(parsed.status, "completed")
        self.assertIsNone(parsed.incomplete_reason)
        self.assertIsNone(parsed.error)


class LifecycleTests(unittest.TestCase):
    def test_warmup_routes_each_model_family_to_its_api(self):
        tester = object.__new__(LatencyTester)
        tester.max_output_tokens = 4096
        tester.configurations = [
            TestConfiguration(
                "gpt-4.1",
                "deployment-41",
                CHAT_COMPLETIONS_API,
            ),
            TestConfiguration(
                "gpt-5.4",
                "deployment-54",
                RESPONSES_API,
                "none",
            ),
        ]
        chat_create = Mock()
        responses_create = Mock()
        tester.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=chat_create),
            ),
            responses=SimpleNamespace(create=responses_create),
        )

        tester.warmup_clients()

        self.assertEqual(chat_create.call_args.kwargs["max_tokens"], 32)
        self.assertEqual(
            responses_create.call_args.kwargs["max_output_tokens"],
            32,
        )
        self.assertEqual(
            responses_create.call_args.kwargs["reasoning"],
            {"effort": "none"},
        )

    def test_measured_gpt_5_response_uses_responses_result_shape(self):
        tester = object.__new__(LatencyTester)
        tester.max_output_tokens = 256
        create = Mock(
            return_value=SimpleNamespace(
                usage=SimpleNamespace(
                    total_tokens=15,
                    output_tokens=9,
                    input_tokens=6,
                    output_tokens_details=SimpleNamespace(
                        reasoning_tokens=3
                    ),
                ),
                output_text="done",
                status="completed",
                incomplete_details=None,
                error=None,
            )
        )
        tester.client = SimpleNamespace(
            responses=SimpleNamespace(create=create),
        )
        configuration = TestConfiguration(
            "gpt-5.6-sol",
            "deployment-56-sol",
            RESPONSES_API,
            "max",
        )

        result = tester.test_model_latency(configuration, "hello")

        self.assertEqual(create.call_args.kwargs["reasoning"], {"effort": "max"})
        self.assertEqual(result["api"], RESPONSES_API)
        self.assertEqual(result["output_tokens"], 9)
        self.assertEqual(result["input_tokens"], 6)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["success"])

    def test_close_releases_client_and_credential(self):
        tester = object.__new__(LatencyTester)
        tester.client = Mock()
        tester.credential = Mock()

        tester.close()

        tester.client.close.assert_called_once_with()
        tester.credential.close.assert_called_once_with()


class CsvTests(unittest.TestCase):
    def test_save_results_uses_stable_schema(self):
        result = {field: None for field in CSV_FIELDS}
        result.update(
            {
                "model": "gpt-5.4",
                "deployment": "deployment-54",
                "success": True,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            LatencyTester.save_results([result], output)
            with output.open(encoding="utf-8", newline="") as input_file:
                reader = csv.DictReader(input_file)
                self.assertEqual(tuple(reader.fieldnames or ()), CSV_FIELDS)
                self.assertEqual(next(reader)["deployment"], "deployment-54")


if __name__ == "__main__":
    unittest.main()
