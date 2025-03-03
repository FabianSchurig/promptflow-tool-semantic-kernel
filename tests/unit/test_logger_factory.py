import logging
from promptflow_tool_semantic_kernel.tools.logger_factory import LoggerFactory


def test_create_logger_default_level():
    logger = LoggerFactory.create_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO


def test_create_logger_custom_level():
    logger = LoggerFactory.create_logger("test_logger", logging.DEBUG)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG
