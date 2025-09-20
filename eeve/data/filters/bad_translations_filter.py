import torch
import torch.nn.functional as F

from openai import OpenAI
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from eeve.utils.datatrove import _get_value


class BadTranslationsFilter(BaseFilter):
    name = "🔄 Bad Translations Cleaner"

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        list_path: list[str],
        sim_score: float,
        normalize: bool = True,
        query_prepocess_fn: None = None,
        passage_prepocess_fn: None = None,
        exclusion_writer: DiskWriter = None
    ):
        super().__init__(exclusion_writer)

        if len(list_path) != 2:
            raise ValueError(f"list_path must contain exactly 2 paths, got {len(list_path)}")
        
        if sim_score < 0 or sim_score > 1:
            raise ValueError(f"sim_score must be between 0 and 1, got {sim_score}")
        
        self.list_path = list_path
        self.sim_score = sim_score
        
        self.client = client
        self.model_name = model_name
        
        self.normailze = normalize
        self.query_prepocess_fn = query_prepocess_fn
        self.passage_prepocess_fn = passage_prepocess_fn

    def filter(self, doc: Document) -> bool:
        raise NotImplementedError

    def filter_batch(self, batch: list[Document]) -> list[bool | tuple[bool, str]]:
        q_batch, p_batch = [], []
        bs = len(batch)
        for doc in batch:
            query, passage = _get_value(doc, self.list_path)

            if self.query_prepocess_fn is not None:
                query = self.query_prepocess_fn(query)
            if self.passage_prepocess_fn is not None:
                passage = self.passage_prepocess_fn(passage)
            q_batch.append(query)
            p_batch.append(passage)
        
        docs = q_batch + p_batch

        response = self.client.embeddings.create(
            model=self.model_name,
            input=docs
        )

        embeddings = torch.tensor([data.embedding for data in response.data])

        if self.normailze:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        q_emb = embeddings[:bs]
        p_emb = embeddings[bs:]
        sims = (q_emb * p_emb).sum(dim=1)

        return (sims > self.sim_score).tolist()