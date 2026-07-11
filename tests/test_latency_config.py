import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from test_latency import (
    CSV_FIELDS,
    TestConfiguration,
    build_completion_params,
    build_model_configs,
    get_completion_error,
    get_reasoning_efforts,
    load_configuration_environment,
    normalize_azure_openai_base_url,
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

    def test_complete_v1_endpoint(self):
        self.assertEqual(
            normalize_azure_openai_base_url(
                "https://example.services.ai.azure.com/openai/v1/"
            ),
            "https://example.services.ai.azure.com/openai/v1/",
        )

    def test_rejects_dated_api_path(self):
        with self.assertRaisesRegex(ValueError, "resource endpoint"):
            normalize_azure_openai_base_url(
                "https://example.openai.azure.com/openai/deployments/demo"
            )


class ModelConfigurationTests(unittest.TestCase):
    def test_default_efforts_use_current_high_to_low_order(self):
        self.assertEqual(
            get_reasoning_efforts({}),
            ["xhigh", "high", "medium", "low", "none"],
        )

    def test_minimal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "minimal"):
            get_reasoning_efforts({"REASONING_EFFORTS": "high,minimal"})

    def test_responses_only_max_effort_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max"):
            get_reasoning_efforts({"REASONING_EFFORTS": "max"})

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
                TestConfiguration("gpt-5.1", "arbitrary-prod-a", "none"),
                TestConfiguration("gpt-5.4", "arbitrary-prod-b", "xhigh"),
                TestConfiguration("gpt-5.4", "arbitrary-prod-b", "none"),
                TestConfiguration("gpt-5.6-sol", "arbitrary-prod-c", "xhigh"),
                TestConfiguration("gpt-5.6-sol", "arbitrary-prod-c", "none"),
            ],
        )

    def test_all_gpt_56_models_support_xhigh_effort(self):
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
                        efforts=["xhigh"],
                    ),
                    [
                        TestConfiguration(
                            model_id,
                            f"{model_id}-deployment",
                            "xhigh",
                        )
                    ],
                )

    def test_gpt_56_chat_effort_matrix(self):
        configurations = build_model_configs(
            {"GPT_56_TERRA_DEPLOYMENT_NAME": "terra-deployment"}
        )
        self.assertEqual(
            [item.reasoning_effort for item in configurations],
            ["xhigh", "high", "medium", "low", "none"],
        )

    def test_legacy_mixed_case_gpt_4o_key_is_supported(self):
        self.assertEqual(
            build_model_configs({"GPT_4o_DEPLOYMENT_NAME": "deployment-4o"}),
            [TestConfiguration("gpt-4o", "deployment-4o")],
        )

    def test_conflicting_gpt_4o_key_spellings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting values"):
            build_model_configs(
                {
                    "GPT_4O_DEPLOYMENT_NAME": "deployment-a",
                    "GPT_4o_DEPLOYMENT_NAME": "deployment-b",
                }
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
    def test_standard_model_uses_max_tokens(self):
        params = build_completion_params(
            TestConfiguration("gpt-4.1", "deployment-41"),
            "hello",
            128,
        )
        self.assertEqual(params["max_tokens"], 128)
        self.assertNotIn("max_completion_tokens", params)
        self.assertNotIn("reasoning_effort", params)

    def test_reasoning_model_uses_max_completion_tokens(self):
        params = build_completion_params(
            TestConfiguration("gpt-5.4", "deployment-54", "xhigh"),
            "hello",
            128,
        )
        self.assertEqual(params["max_completion_tokens"], 128)
        self.assertEqual(params["reasoning_effort"], "xhigh")
        self.assertNotIn("max_tokens", params)


class CompletionValidationTests(unittest.TestCase):
    def test_complete_text_is_accepted(self):
        self.assertIsNone(get_completion_error("stop", "완료"))

    def test_length_truncation_is_rejected(self):
        self.assertIn(
            "MAX_OUTPUT_TOKENS",
            get_completion_error("length", "") or "",
        )

    def test_empty_text_is_rejected(self):
        self.assertEqual(
            get_completion_error("stop", "  "),
            "The model returned an empty response.",
        )

    def test_content_filter_has_specific_error(self):
        self.assertIn(
            "content filter",
            (get_completion_error("content_filter", "") or "").lower(),
        )


class LifecycleTests(unittest.TestCase):
    def test_warmup_accepts_truncated_response(self):
        tester = object.__new__(LatencyTester)
        tester.max_output_tokens = 4096
        tester.configurations = [
            TestConfiguration("gpt-5.4", "deployment-54", "high")
        ]
        create = Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content=None),
                    )
                ]
            )
        )
        tester.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )

        tester.warmup_clients()

        self.assertEqual(create.call_args.kwargs["max_completion_tokens"], 32)

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
