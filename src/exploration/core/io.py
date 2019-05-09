import logging
import sys
from logging import Logger

from pyspark.sql import DataFrame, SQLContext, SparkSession


def get_logger() -> Logger:
    logger = logging.getLogger("X.CNAM")
    if len(logger.handlers) == 0:
        logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(screen_handler)
    return logger


def quiet_spark_logger(sc: SparkSession) -> Logger:
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    return logger


def get_spark_context():
    return SQLContext.getOrCreate(SparkSession.builder.getOrCreate())


def read_data_frame(filepath: str) -> DataFrame:
    return get_spark_context().read.parquet(filepath)
