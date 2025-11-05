# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger
import logging

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import rotate_model, hadamard_input_projections_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from utils.range_utils import (
    OutlierStatsCollector,
    log_outlier_stats,
    RangeCollector,
    RangeCollectorPerDim,
    HistogramAccumulator,
    plot_histograms,
    save_histogram_summary,
    log_histogram_summary,
    save_histograms_bincount,
)

log: Logger = utils.get_logger("spinquant")


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    # Set up file logging
    log_file = f"rotate_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    )
    log.addHandler(file_handler)

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()

    model = rotate_model(ptq_args, model, model_args)
    model = hadamard_input_projections_model(ptq_args, model, model_args)

    print(model)

    log.info("rotate model completed")

    range_collector = RangeCollector()
    range_collector_per_dim = RangeCollectorPerDim()
    range_collector.register_hooks(model)  # Re-enabled for histogram collection
    range_collector_per_dim.register_hooks(model)
    # Initialize KLL tail statistics collector
    # from utils.kll_collector import KLLTailCollector
    # kll_collector = KLLTailCollector(alpha=0.999)
    # kll_collector.register_hooks(model)

    # outlier_stats_collector = OutlierStatsCollector(threshold_r=10.0)
    # outlier_stats_collector.register_hooks(model)
    model.seqlen = training_args.model_max_length

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))

    # Complete KLL tail stats collection and analysis
    range_collector.remove_hooks()
    range_collector_per_dim.remove_hooks()

    from utils.range_utils import log_filtered_ranges, log_filtered_ranges_summary

    filtered_ranges_4 = range_collector_per_dim.get_per_dim_ranges_filtered_cpu(
        threshold=4
    )
    filtered_ranges_8 = range_collector_per_dim.get_per_dim_ranges_filtered_cpu(
        threshold=8
    )
    filtered_ranges_16 = range_collector_per_dim.get_per_dim_ranges_filtered_cpu(
        threshold=16
    )
    filtered_ranges_32 = range_collector_per_dim.get_per_dim_ranges_filtered_cpu(
        threshold=32
    )

    log_filtered_ranges(filtered_ranges_4, "Filtered ranges for 4", logger=log)
    log.info("Filtered ranges for 4, 8, 16, 32:")
    log_filtered_ranges_summary(filtered_ranges_4, logger=log)
    log_filtered_ranges_summary(filtered_ranges_8, logger=log)
    log_filtered_ranges_summary(filtered_ranges_16, logger=log)
    log_filtered_ranges_summary(filtered_ranges_32, logger=log)

    # kll_collector.complete_collection(log, local_rank)

    # testloader = data_utils.get_wikitext2(
    #     seed=ptq_args.seed,
    #     seqlen=2048,
    #     tokenizer=tokenizer,
    #     eval_mode=True,
    # )
    # histogram_accumulator = HistogramAccumulator(range_collector.get_min_max_ranges())
    # histogram_accumulator.register_hooks(model)
    # _ = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    # histograms = histogram_accumulator.get_cpu_histograms()
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # plot_histograms(
    #     histograms,
    #     histogram_accumulator.layer_ranges,
    #     output_dir="histograms/" + timestamp,
    # )
    # save_histograms_bincount(
    #     histograms,
    #     histogram_accumulator.layer_ranges,
    #     output_dir=f"histograms_bincount_{timestamp}",
    # )

    # save_histogram_summary(
    #     histograms,
    #     histogram_accumulator.layer_ranges,
    #     output_file=f"activation_summary_{timestamp}.txt",
    # )
    # log_histogram_summary(log, histograms, histogram_accumulator.layer_ranges)

    dist.barrier()


if __name__ == "__main__":
    train()
