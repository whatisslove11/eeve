import os

import yaml
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import (
    SentenceDedupFilter,
    SentenceDedupSignature,
    SentenceFindDedups,
)
from datatrove.pipeline.dedup.sentence_dedup import SentDedupConfig
from datatrove.pipeline.tokens import TokensCounter

from eeve.utils.datatrove import create_reader_and_writer


class DeduplicationPaths:
    def __init__(self, base_data_dir, base_logging_dir):
        self.signatures_dir = os.path.join(base_data_dir, "sigs")
        self.duplicates_dir = os.path.join(base_data_dir, "dups")

        if base_logging_dir is not None:
            self.logging_signatures_dir = os.path.join(base_logging_dir, "signatures")
            self.logging_buckets_dir = os.path.join(base_logging_dir, "buckets")
            self.logging_clusters_dir = os.path.join(base_logging_dir, "clusters")
        else:
            self.logging_signatures_dir = None
            self.logging_buckets_dir = None
            self.logging_clusters_dir = None


def sentence_deduplication(
    INPUT_READER,
    OUTPUT_WRITER,
    sent_dedup_config: dict,
    paths: DeduplicationPaths,
    language: str,
    total_tasks: int,
    workers: int,
):
    sent_dedup_config = SentDedupConfig(**sent_dedup_config)

    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            SentenceDedupSignature(
                output_folder=paths.signatures_dir,
                config=sent_dedup_config,
                finder_workers=workers,
                language=language,
            ),
        ],
        logging_dir=paths.logging_signatures_dir,
        tasks=total_tasks,
        workers=workers,
    )

    stage2 = LocalPipelineExecutor(
        pipeline=[
            SentenceFindDedups(
                data_folder=paths.signatures_dir,
                output_folder=paths.duplicates_dir,
                config=sent_dedup_config,
            ),
        ],
        depends=stage1,
        logging_dir=paths.logging_buckets_dir,
        tasks=total_tasks,
        workers=1,
    )

    stage3 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(),
            SentenceDedupFilter(
                data_folder=paths.duplicates_dir,
                config=sent_dedup_config,
                language=language,
            ),
            OUTPUT_WRITER,
        ],
        depends=stage2,
        logging_dir=paths.logging_clusters_dir,
        tasks=total_tasks,
        workers=workers,
    )

    stage3.run()


def run_sentence_deduplication(path_to_yaml_config: str) -> None:
    with open(path_to_yaml_config, "r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f) or {}

    if not yaml_config:
        raise ValueError("Config is empty or incorrect.")

    INPUT_READER, OUTPUT_WRITER = create_reader_and_writer(yaml_config)

    paths = DeduplicationPaths(
        base_data_dir=yaml_config["base_data_dir"],
        base_logging_dir=yaml_config["base_logging_dir"],
    )

    sentence_deduplication(
        INPUT_READER=INPUT_READER,
        OUTPUT_WRITER=OUTPUT_WRITER,
        sent_dedup_config=yaml_config["dedup_config"],
        paths=paths,
        language=yaml_config["language"],
        total_tasks=yaml_config["num_shards"],
        workers=yaml_config["workers"],
    )
