from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.writers.huggingface import HuggingFaceDatasetWriter
from datatrove.pipeline.tokens import TokensCounter
from datatrove.utils.hashing import HashConfig


def minhash_dedup(
    INPUT_READER,
    minhash_base_path: str,
    language: str,
    num_buckets: int,
    hashes_per_bucket: int,
    upload_path: str,
    local_working_dir: str | None = None,
    base_logging_dir: str | None = None,
    total_tasks: int = 1,
    workers: int = -1
):
    minhash_config = MinhashConfig(
        hash_config=HashConfig(
            precision=64,
            hash_fc="sha1"
        ),
        num_buckets=num_buckets,
        hashes_per_bucket=hashes_per_bucket,
        n_grams=5,
    )

    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=f"{minhash_base_path}/signatures",
                config=minhash_config,
                language=language
            ),
        ],
        logging_dir=f'{base_logging_dir}/signatures',
        tasks=total_tasks,
        workers=workers
    )

    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{minhash_base_path}/signatures",
                output_folder=f"{minhash_base_path}/buckets",
                config=minhash_config,
            ),
        ],
        depends=stage1,
        logging_dir=f'{base_logging_dir}/buckets',
        tasks=minhash_config.num_buckets,
        workers=workers
    )

    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{minhash_base_path}/buckets",
                output_folder=f"{minhash_base_path}/remove_ids",
                config=minhash_config,
            ),
        ],
        depends=stage2,
        logging_dir=f'{base_logging_dir}/clusters',
        tasks=minhash_config.num_buckets,
        workers=workers
    )

    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(),  
            MinhashDedupFilter(
                input_folder=f"{minhash_base_path}/remove_ids",
                exclusion_writer=JsonlWriter(f"{minhash_base_path}/removed"),
            ),
            HuggingFaceDatasetWriter(
                dataset=upload_path,
                private=True,
                local_working_dir=local_working_dir, # необяз арг
                output_filename="data/${rank}.parquet",
                cleanup=True, # нужно ли удалять с локальной папки датасет потом
            ),
        ],
        depends=stage3,
        logging_dir=f'{base_logging_dir}/clusters',
        tasks=minhash_config.num_buckets,
        workers=workers
    )

    stage4.run()
