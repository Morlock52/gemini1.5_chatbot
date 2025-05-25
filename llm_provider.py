import os
from typing import List

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - library may not be installed
    genai = None

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
except ImportError:  # pragma: no cover - library may not be installed
    vertexai = None
    GenerativeModel = None

try:
    import openai
except ImportError:  # pragma: no cover - library may not be installed
    openai = None


class LLMProvider:
    """Simple abstraction over different LLM service providers."""

    def __init__(self, provider: str, model_name: str, **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.kwargs = kwargs
        self._init_model()

    def _init_model(self) -> None:
        if self.provider == "google" and genai:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.kwargs.get("generation_config"),
            )
        elif self.provider == "vertex" and vertexai and GenerativeModel:
            projectid = os.environ.get("GOOG_PROJECT")
            if projectid:
                vertexai.init(project=projectid, location="us-central1")
            self.model = GenerativeModel(
                model_name=self.model_name,
                generation_config=self.kwargs.get("generation_config"),
            )
        elif self.provider == "openai" and openai:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            self.temperature = self.kwargs.get("temperature", 1.0)
            self.max_tokens = self.kwargs.get("max_tokens", 2000)
        else:
            raise ValueError(f"Unsupported provider or missing dependencies: {self.provider}")

    def generate_content(self, prompts: List[str]) -> str:
        """Generate content using the chosen provider."""
        if self.provider in {"google", "vertex"}:
            response = self.model.generate_content(prompts)
            return getattr(response, "text", str(response))
        elif self.provider == "openai":
            if not openai:
                raise RuntimeError("openai package is not available")
            messages = [{"role": "user", "content": p} for p in prompts]
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
