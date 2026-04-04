"""
Azure OpenAI GPT Models Latency Comparison Test
Tests GPT-4.1 and GPT-5.4 with reasoning effort
Uses Azure OpenAI SDK with chat.completions.create()
"""

import os
import time
from typing import Dict, List
from datetime import datetime
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

# Load environment variables
load_dotenv()





class LatencyTester:
    def __init__(self):
        # Azure OpenAI 클라이언트 생성 (모든 모델 공용)
        endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
        
        if not endpoint:
            raise ValueError("AZURE_AI_PROJECT_ENDPOINT is required")
        
        # Extract base endpoint (remove /api/projects/... part)
        if "/api/projects/" in endpoint:
            base_endpoint = endpoint.split("/api/projects/")[0]
        else:
            base_endpoint = endpoint
            
        print(f"Using Azure OpenAI SDK with DefaultAzureCredential")
        print(f"Endpoint: {base_endpoint}")
        
        # DefaultAzureCredential 기반 토큰 인증 (az login 필요)
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        
        # Create a single Azure OpenAI client for all models
        self.client = AzureOpenAI(
            azure_ad_token_provider=token_provider,
            api_version="2024-08-01-preview",
            azure_endpoint=base_endpoint
        )
        
        print(f"API Version: 2024-08-01-preview\n")
        
        # Model deployment names - 모든 배포된 모델 및 reasoning effort 조합
        self.models = {
            # GPT-4 시리즈 (non-reasoning models)
            "gpt-4.1": os.getenv("GPT_41_DEPLOYMENT_NAME", "gpt-4.1"),

            # GPT-5.4 시리즈 - reasoning_effort 파라미터 지원 (low/medium/high/minimal)
            #"gpt-5.4-high": (os.getenv("GPT_54_DEPLOYMENT_NAME", "gpt-5.4"), "high"),
            #"gpt-5.4-medium": (os.getenv("GPT_54_DEPLOYMENT_NAME", "gpt-5.4"), "medium"),
            #"gpt-5.4-low": (os.getenv("GPT_54_DEPLOYMENT_NAME", "gpt-5.4"), "low"),
            "gpt-5.4-minimal": (os.getenv("GPT_54_DEPLOYMENT_NAME", "gpt-5.4"), "minimal"),
            #"gpt-5.4-none": (os.getenv("GPT_54_DEPLOYMENT_NAME", "gpt-5.4"), "none"),
            #"gpt-5.4-default": os.getenv("GPT_54_DEPLOYMENT_NAME", "gpt-5.4"),
            
            #"gpt-5.4-mini-high": (os.getenv("GPT_54_MINI_DEPLOYMENT_NAME", "gpt-5.4-mini"), "high"),
            #"gpt-5.4-mini-medium": (os.getenv("GPT_54_MINI_DEPLOYMENT_NAME", "gpt-5.4-mini"), "medium"),
            #"gpt-5.4-mini-low": (os.getenv("GPT_54_MINI_DEPLOYMENT_NAME", "gpt-5.4-mini"), "low"),
            "gpt-5.4-mini-minimal": (os.getenv("GPT_54_MINI_DEPLOYMENT_NAME", "gpt-5.4-mini"), "minimal"),
            #"gpt-5.4-mini-none": (os.getenv("GPT_54_MINI_DEPLOYMENT_NAME", "gpt-5.4-mini"), "none"),
            "gpt-5.4-mini-default": os.getenv("GPT_54_MINI_DEPLOYMENT_NAME", "gpt-5.4-mini"),
        }
        
        # Test prompts - varied complexity (5 prompts for comprehensive testing)
        self.test_prompts = [
            "프랑스의 수도는 어디인가요?",
            "양자 컴퓨팅을 쉽게 설명해주세요.",
            "팩토리얼을 계산하는 파이썬 함수를 작성해주세요."
        ]
    
    def test_model_latency(self, model_name: str, deployment_info, prompt: str) -> Dict:
        """Test latency for a single model with a single prompt"""
        try:
            start_time = time.time()
            
            # deployment_info가 튜플이면 (deployment_name, reasoning_effort) 형태
            if isinstance(deployment_info, tuple):
                deployment_name, reasoning_effort = deployment_info
            else:
                deployment_name = deployment_info
                reasoning_effort = None
            
            client = self.client
            
            # Prepare chat completion parameters
            messages = [
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Create chat completion with reasoning_effort parameter
            params = {
                "model": deployment_name,
                "messages": messages
            }
            
            # Add reasoning_effort parameter for GPT-5/5.4 models
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            
            # Create chat completion - no fallback, let errors surface
            response = client.chat.completions.create(**params)
            
            # Print reasoning_effort info only when 'none' is requested
            if reasoning_effort == "none":
                if hasattr(response, 'usage') and hasattr(response.usage, 'reasoning_tokens'):
                    print(f"  Reasoning tokens with 'none': {response.usage.reasoning_tokens}")
                
                # Print actual reasoning_effort value from response if available
                if hasattr(response, 'reasoning_effort'):
                    print(f"  Response reasoning_effort: {response.reasoning_effort}")
                elif hasattr(response.choices[0].message, 'reasoning_effort'):
                    print(f"  Message reasoning_effort: {response.choices[0].message.reasoning_effort}")
                else:
                    # Check if it's in the response dict
                    response_dict = response.model_dump() if hasattr(response, 'model_dump') else {}
                    if 'reasoning_effort' in response_dict:
                        print(f"  Default reasoning_effort from response: {response_dict['reasoning_effort']}")
                    else:
                        print(f"  No reasoning_effort field found in response")
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Extract response
            response_content = response.choices[0].message.content if response.choices else ""
            
            return {
                "model": model_name,
                "prompt": prompt,
                "latency_ms": round(latency, 2),
                "tokens": response.usage.total_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "response": response_content[:100],
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "model": model_name,
                "prompt": prompt,
                "latency_ms": None,
                "tokens": None,
                "completion_tokens": None,
                "prompt_tokens": None,
                "response": None,
                "success": False,
                "error": str(e)
            }
    
    def warmup_clients(self):
        """Warm up API clients by making dummy calls to establish connections"""
        print(f"\n{'='*80}")
        print("Warming up API connections...")
        print(f"{'='*80}\n")
        
        warmup_prompt = "Hi"
        warmup_models = set()
        
        # Collect unique deployment names to warm up
        for deployment_info in self.models.values():
            if isinstance(deployment_info, tuple):
                deployment_name = deployment_info[0]
            else:
                deployment_name = deployment_info
            warmup_models.add(deployment_name)
        
        for deployment_name in warmup_models:
            try:
                client = self.client
                
                print(f"Warming up {deployment_name}...", end=" ")
                start = time.time()
                
                # GPT-5 and GPT-5.1 models use max_completion_tokens instead of max_tokens
                # Increase token limit to avoid truncation errors
                if "gpt-5" in deployment_name.lower():
                    response = client.chat.completions.create(
                        model=deployment_name,
                        messages=[{"role": "user", "content": warmup_prompt}],
                        max_completion_tokens=50
                    )
                else:
                    response = client.chat.completions.create(
                        model=deployment_name,
                        messages=[{"role": "user", "content": warmup_prompt}],
                        max_tokens=50
                    )
                
                elapsed = (time.time() - start) * 1000
                print(f"{elapsed:.0f}ms")
                
            except Exception as e:
                print(f"Failed: {e}")
        
        print(f"\nWarmup complete!\n")
    
    def run_tests(self, iterations: int = 1) -> List[Dict]:
        """Run latency tests for all models"""
        results = []

        print(f"\n{'='*80}")
        print(f"Azure OpenAI Latency Comparison Test")
        print(f"Testing {len(self.models)} model configurations")
        print(f"Iterations per prompt: {iterations}")
        print(f"{'='*80}\n")

        for model_name, deployment_info in self.models.items():
            print(f"\n{'='*80}")
            print(f"Testing: {model_name}")
            print(f"{'='*80}")
            
            for i, prompt in enumerate(self.test_prompts, 1):
                print(f"\n📝 Prompt {i}/{len(self.test_prompts)}: {prompt}")
                
                for iteration in range(iterations):
                    result = self.test_model_latency(model_name, deployment_info, prompt)
                    result["iteration"] = iteration + 1
                    result["timestamp"] = datetime.now().isoformat()
                    results.append(result)
                    
                    if result["success"]:
                        print(f"\n💬 Response: {result['response']}")
                        print(f"⏱️  Latency: {result['latency_ms']:.0f}ms")
                        print(f"🔢 Tokens - Total: {result['tokens']}, Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}")
                    else:
                        print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
                    
                    time.sleep(0.2)

        return results
    
    def analyze_results(self, results: List[Dict]):
        """Analyze and display results with comprehensive statistics"""
        print(f"\n{'='*80}")
        print("DETAILED RESULTS - PROMPTS AND RESPONSES")
        print(f"{'='*80}\n")
        
        # Display all prompts and responses with token usage
        for i, result in enumerate(results, 1):
            if result["success"]:
                print(f"Test {i}: {result['model']}")
                print(f"Prompt: {result['prompt']}")
                print(f"Response: {result['response']}")
                print(f"Latency: {result['latency_ms']:.0f}ms")
                print(f"Tokens - Total: {result['tokens']}, Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}")
                print("-" * 80)
        
        print(f"\n{'='*80}")
        print("LATENCY ANALYSIS")
        print(f"{'='*80}\n")
        
        # Group by model and prompt
        model_stats = {}
        model_prompt_stats = {}
        model_token_stats = {}
        
        for result in results:
            if result["success"]:
                model = result["model"]
                prompt = result["prompt"]
                
                if model not in model_stats:
                    model_stats[model] = {"latencies": []}
                    model_prompt_stats[model] = {}
                    model_token_stats[model] = {"total": [], "prompt": [], "completion": []}
                
                model_stats[model]["latencies"].append(result["latency_ms"])
                model_token_stats[model]["total"].append(result["tokens"])
                model_token_stats[model]["prompt"].append(result["prompt_tokens"])
                model_token_stats[model]["completion"].append(result["completion_tokens"])
                
                if prompt not in model_prompt_stats[model]:
                    model_prompt_stats[model][prompt] = []
                model_prompt_stats[model][prompt].append(result["latency_ms"])
        
        # Display detailed per-prompt latencies
        print("DETAILED LATENCY BY PROMPT")
        print("-" * 120)
        
        sorted_models = sorted(model_stats.items(), key=lambda x: sum(x[1]["latencies"])/len(x[1]["latencies"]))
        
        for model, stats in sorted_models:
            print(f"\n{model}:")
            for i, prompt in enumerate(self.test_prompts, 1):
                if prompt in model_prompt_stats[model]:
                    latencies = model_prompt_stats[model][prompt]
                    avg = sum(latencies) / len(latencies)
                    print(f"  Prompt {i}: {avg:.0f}ms (iterations: {latencies})")
        
        # Calculate overall statistics
        print(f"\n{'='*80}")
        print("OVERALL STATISTICS")
        print(f"{'='*80}\n")
        print(f"{'Model':<30} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Std Dev':<12} {'Avg Tokens':<12} {'Tests':<8}")
        print("-" * 110)
        
        for model, stats in sorted_models:
            latencies = stats["latencies"]
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            # Calculate standard deviation
            variance = sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)
            std_dev = variance ** 0.5
            
            # Calculate average tokens
            avg_tokens = sum(model_token_stats[model]["total"]) / len(model_token_stats[model]["total"])
            
            print(f"{model:<30} {avg_latency:<12.2f} {min_latency:<12.2f} {max_latency:<12.2f} {std_dev:<12.2f} {avg_tokens:<12.1f} {len(latencies):<8}")
        
        # Error summary
        errors = [r for r in results if not r["success"]]
        if errors:
            print(f"\n{'='*80}")
            print(f"ERRORS ({len(errors)} total)")
            print(f"{'='*80}\n")
            for error in errors:
                print(f"Model: {error['model']}")
                print(f"Error: {error['error']}\n")
    
    def save_results(self, results: List[Dict], filename: str = "latency_results.csv"):
        """Save results to CSV file"""
        import csv
        
        if not results:
            print("No results to save.")
            return
        
        keys = results[0].keys()
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to {filename}")


def main():
    # Check environment variables
    endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    
    if not endpoint:
        print("Error: AZURE_AI_PROJECT_ENDPOINT is required")
        print("Set it in your .env file")
        return
    
    print("Using Azure OpenAI SDK with DefaultAzureCredential...")
    
    try:
        # Run tests
        tester = LatencyTester()
        
        # Warm up connections first
        tester.warmup_clients()
        
        results = tester.run_tests(iterations=1)
        
        # Analyze results
        tester.analyze_results(results)
        
        # Save results
        tester.save_results(results)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
