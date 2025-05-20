from eamld.logging_config import setup_logger
import logging
from eamld.contraction_strategy.hypergraph_to_connectivity import ConnectivityGraph

if __name__ == "__main__":

    # 调用 setup_logger 函数并获取一个日志记录器
    logger = setup_logger(__name__, logging.DEBUG)

    graph = ConnectivityGraph()
    
    # 测试日志记录
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")