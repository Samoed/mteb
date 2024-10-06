from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader
from sentence_transformers import SentenceTransformer


class JinaWrapper:
    """following the hf model card documentation."""
    # todo install einops, sbert > 3.1.0
    original_prompts = {
        "retrieval.query":"Represent the query for retrieving evidence documents: ",
        "retrieval.passage":"Represent the document for retrieval: ",
        "separation": "",
        "classification": "",
        "text-matching": ""
    }

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
        task = self.prompts.get(prompt_name, "")
        prompt = self.original_prompts.get(task, "")

        return self.model.encode(
            sentences,
            batch_size=batch_size,
            task=task,  # special jina parameter
            prompt=prompt,
            **kwargs
        )


jina_emdeddings_v3 = ModelMeta(
    loader=partial(
        JinaWrapper,
        model_name="jinaai/jina-embeddings-v3",
        revision="343dbf534c76fe845f304fa5c2d1fd87e1e78918",
        trust_remote_code=True,
        prompts={
            "Retrieval-query": "retrieval.query",
            "Retrieval-passage": "retrieval.passage",
            "Clustering": "separation",
            "Classification": "classification",
            "STS": "text-matching",
            "PairClassification": "text-matching",
            "BitextMining": "text-matching",
            "MultiLabelClassification": "classification",
            "Reranking": "separation", # todo test
            "Summarization": "text-matching",
        },
    ),
    name="jinaai/jina-embeddings-v3",
    languages=["eng_Latn"],
    open_source=True,
    revision="343dbf534c76fe845f304fa5c2d1fd87e1e78918",
    release_date="2024-09-05",  # initial commit of hf model.
)
