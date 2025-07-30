import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def clean_mem():
    import gc
    import os
    import sys
    import time
    import traceback

    import psutil
    import torch

    process = psutil.Process(os.getpid())

    # Measure RAM before cleanup
    ram_before = process.memory_info().rss / (1024**2)  # in MB

    # Measure GPU before cleanup
    if torch.cuda.is_available():
        gpu_alloc_before = torch.cuda.memory_allocated() / (1024**2)  # in MB
        gpu_reserved_before = torch.cuda.memory_reserved() / (1024**2)  # in MB
    else:
        gpu_alloc_before = gpu_reserved_before = 0

    # clean all traceback
    if hasattr(sys, "last_traceback"):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, "last_traceback")
    if hasattr(sys, "last_type"):
        delattr(sys, "last_type")
    if hasattr(sys, "last_value"):
        delattr(sys, "last_value")

    # clean all ipython history
    if "get_ipython" in globals():
        try:
            from IPython import get_ipython

            ip = get_ipython()
            user_ns = ip.user_ns
            ip.displayhook.flush()
            pc = ip.displayhook.prompt_count + 1
            for n in range(1, pc):
                user_ns.pop("_i" + repr(n), None)
            user_ns.update(dict(_i="", _ii="", _iii=""))
            hm = ip.history_manager
            hm.input_hist_parsed[:] = [""] * pc
            hm.input_hist_raw[:] = [""] * pc
            hm._i = hm._ii = hm._iii = hm._i00 = ""
        except Exception as e:
            print("ipython mem could not be cleared")

    # do a garbage collection and flush cuda cache
    gc.collect()
    torch.cuda.empty_cache()

    # Give system a small moment to settle (helps RAM measurement be more accurate)
    time.sleep(0.1)

    # Measure RAM after cleanup
    ram_after = process.memory_info().rss / (1024**2)  # in MB

    # Measure GPU after cleanup
    if torch.cuda.is_available():
        gpu_alloc_after = torch.cuda.memory_allocated() / (1024**2)  # in MB
        gpu_reserved_after = torch.cuda.memory_reserved() / (1024**2)  # in MB
    else:
        gpu_alloc_after = gpu_reserved_after = 0

    # Report freed memory
    print(
        f"RAM freed: {ram_before - ram_after:.2f} MB ({ram_before:.2f} -> {ram_after:.2f})"
    )
    if torch.cuda.is_available():
        print(
            f"GPU allocated freed: {gpu_alloc_before - gpu_alloc_after:.2f} MB ({gpu_alloc_before:.2f} -> {gpu_alloc_after:.2f})"
        )
        print(
            f"GPU reserved freed: {gpu_reserved_before - gpu_reserved_after:.2f} MB ({gpu_reserved_before:.2f} -> {gpu_reserved_after:.2f})"
        )
    else:
        print("No GPU detected.")


def create_logger(
    name: str = "reddit_moderation",
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = "logs",
    console_output: bool = True,
    file_output: bool = True,
    format_string: Optional[str] = None,
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5,
    include_timestamp_in_filename: bool = True,
) -> logging.Logger:
    """
    Create a fully featured logger for the Reddit comment moderation system.

    This logger is designed to handle all aspects of the multi-stage classification
    pipeline including zero-shot classification, fine-tuning, and evaluation.

    Parameters
    ----------
    name : str, optional
        Logger name, by default "reddit_moderation"
    log_level : str, optional
        Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        by default "INFO"
    log_file : str or Path, optional
        Specific log file path. If None, auto-generates based on name and timestamp
    log_dir : str or Path, optional
        Directory for log files, by default "logs"
    console_output : bool, optional
        Whether to output logs to console, by default True
    file_output : bool, optional
        Whether to output logs to file, by default True
    format_string : str, optional
        Custom log format string, by default None (uses comprehensive format)
    max_bytes : int, optional
        Maximum log file size before rotation, by default 10MB
    backup_count : int, optional
        Number of backup log files to keep, by default 5
    include_timestamp_in_filename : bool, optional
        Whether to include timestamp in log filename, by default True

    Returns
    -------
    logging.Logger
        Configured logger instance ready for use

    Examples
    --------
    >>> # Basic usage
    >>> logger = create_logger()
    >>> logger.info("Starting Reddit comment classification pipeline")

    >>> # Advanced usage for training
    >>> training_logger = create_logger(
    ...     name="distilbert_training",
    ...     log_level="DEBUG",
    ...     log_file="training_session.log"
    ... )
    >>> training_logger.debug("Training batch processed")

    >>> # For evaluation only
    >>> eval_logger = create_logger(
    ...     name="model_evaluation",
    ...     console_output=False,
    ...     log_file="evaluation_results.log"
    ... )
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplication
    logger.handlers.clear()

    # Default comprehensive format for ML workflows
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if file_output:
        # Create log directory
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)

        # Generate log filename
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if include_timestamp_in_filename:
                log_filename = f"{name}_{timestamp}.log"
            else:
                log_filename = f"{name}.log"
            log_file = log_dir / log_filename if log_dir else Path(log_filename)
        else:
            log_file = Path(log_file)
            if log_dir and not log_file.is_absolute():
                log_file = Path(log_dir) / log_file

        # Create rotating file handler
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add some useful methods to the logger
    def log_dataset_info(dataset, dataset_name="Dataset"):
        """Log dataset information"""
        logger.info(f"{dataset_name} Info:")
        logger.info(f"  - Size: {len(dataset):,} samples")
        logger.info(f"  - Columns: {dataset.column_names}")
        if "labels" in dataset.column_names:
            import numpy as np

            labels = np.array(dataset["labels"])
            unique, counts = np.unique(labels, return_counts=True)
            logger.info(f"  - Label distribution: {dict(zip(unique, counts))}")

    def log_model_info(model, model_name="Model"):
        """Log model information"""
        logger.info(f"{model_name} Info:")
        if hasattr(model, "config"):
            logger.info(f"  - Model type: {model.config.model_type}")
            logger.info(f"  - Hidden size: {model.config.hidden_size}")
            if hasattr(model.config, "num_labels"):
                logger.info(f"  - Number of labels: {model.config.num_labels}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")

    def log_training_args(training_args):
        """Log training arguments"""
        logger.info("Training Configuration:")
        logger.info(f"  - Learning rate: {training_args.learning_rate}")
        logger.info(f"  - Batch size: {training_args.per_device_train_batch_size}")
        logger.info(
            f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}"
        )
        logger.info(f"  - Epochs: {training_args.num_train_epochs}")
        logger.info(f"  - Weight decay: {training_args.weight_decay}")
        logger.info(f"  - LR scheduler: {training_args.lr_scheduler_type}")
        logger.info(f"  - Warmup ratio: {training_args.warmup_ratio}")

    def log_metrics(metrics, stage=""):
        """Log evaluation metrics"""
        stage_prefix = f"{stage} " if stage else ""
        logger.info(f"{stage_prefix}Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  - {metric}: {value:.4f}")
            else:
                logger.info(f"  - {metric}: {value}")

    # Attach utility methods to logger
    logger.log_dataset_info = log_dataset_info
    logger.log_model_info = log_model_info
    logger.log_training_args = log_training_args
    logger.log_metrics = log_metrics

    # Log logger creation
    logger.info(f"Logger '{name}' created successfully")
    logger.info(f"Log level: {log_level}")
    if file_output:
        logger.info(f"Log file: {log_file}")

    return logger


# Convenience function for quick setup
def setup_project_logging(debug_mode: bool = False) -> logging.Logger:
    """
    Quick setup for the Reddit moderation project logging.

    Parameters
    ----------
    debug_mode : bool
        If True, sets log level to DEBUG and enables verbose logging

    Returns
    -------
    logging.Logger
        Configured project logger
    """
    log_level = "DEBUG" if debug_mode else "INFO"

    return create_logger(
        name="reddit_moderation_pipeline",
        log_level=log_level,
        log_dir="project_logs",
        include_timestamp_in_filename=True,
    )
