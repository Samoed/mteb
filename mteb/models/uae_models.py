from __future__ import annotations

from functools import partial
from typing import Any

from sentence_transformers import SentenceTransformer

from mteb.model_meta import ModelMeta


class UAEWrapper:
    """following the hf model card documentation."""

    def __init__(self, model_name: str, revision: str, **kwargs: Any):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.prompts = self.model.prompts

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        prompt_name: str | None = None,
        batch_size: int = 32,
        input_type: str | None = None,
        **kwargs: Any,
    ):
        if prompt_name and prompt_name in self.model.prompts:
            prompt = self.model.prompts[prompt_name]
            sentences = [prompt.format(text=sentence) for sentence in sentences]

        return self.model.encode(
            sentences,
            batch_size=batch_size,
            **kwargs,
        )


uae_large_v1 = ModelMeta(
    loader=partial(
        UAEWrapper,
        model_name="WhereIsAI/UAE-Large-V1",
        revision="369c368f70f16a613f19f5598d4f12d9f44235d4",
        trust_remote_code=True,
        # https://github.com/SeanLee97/AnglE/blob/b04eae166d8596b47293c75b4664d3ad820d7331/angle_emb/angle.py#L291-L314
        prompts={
            "query": "'Represent this sentence for searching relevant passages: {text}'",
            "Summarization": 'Summarize sentence "{text}" in one word:"',
        },
    ),
    name="WhereIsAI/UAE-Large-V1",
    languages=["eng_Latn"],
    open_source=True,
    revision="369c368f70f16a613f19f5598d4f12d9f44235d4",
    release_date="2023-12-04",  # initial commit of hf model.
)
