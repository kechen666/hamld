import logging

# config.py
import os

def setup_logger(name: str, log_level: int = logging.WARNING, log_type: str = 'console') -> logging.Logger:
    """
    设置并返回一个配置好的日志记录器，适用于不同环境配置。

    参数:
    - name: 日志记录器的名称，通常使用 __name__，表示当前模块的名称
    - log_level: logging.DEBUG, logging.INFO, logging.NOTSET
    - log_type: 日志输出类型 ('console', 'file', 'both')，默认为 'console'
    

    返回:
    - logger: 配置好的日志记录器
    """
    # 获取日志记录器
    logger = logging.getLogger(name)

    # 设置日志级别为DEBUG，确保能够捕获DEBUG及以上级别的日志
    logger.setLevel(log_level)

    # 设置日志输出格式（时间戳、日志级别、消息）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)



    # 根据 log_type 来选择处理器
    if log_type == 'console' or log_type == 'both':
        logger.addHandler(console_handler)
    elif log_type == 'file' or log_type == 'both':
        # 如果是文件类型，只添加文件处理器
        os.makedirs('logs', exist_ok=True)  # 创建日志目录
        
        # 文件日志处理器
        file_handler = logging.FileHandler(f"logs/{name}.log", mode='w')  # 覆盖写入日志文件
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# def setup_logging(environment='development', log_file_name="surface_code.log"):
#     """
#     根据环境设置日志配置。
#     - 开发环境（development）输出到控制台和日志文件。
#     - 生产环境（production）输出到日志文件，日志文件使用追加模式。

#     参数:
#     - environment: 配置环境 ('development' 或 'production')
#     - log_file_name: 日志文件名
#     """
#     # 创建 logs 目录（如果不存在的话）
#     os.makedirs('logs', exist_ok=True)

#     # 配置日志
#     if environment == 'development':
#         logging.basicConfig(
#             level=logging.DEBUG,
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.StreamHandler(),  # 输出到控制台
#                 logging.FileHandler(f"logs/{log_file_name}", mode='w')  # 覆盖写入日志文件
#             ]
#         )
#     else:
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.FileHandler(f"logs/{log_file_name}", mode='a')  # 追加模式
#             ]
#         )

# 示例：调用封装的函数来获取日志记录器并使用
if __name__ == "__main__":

    # 调用 setup_logger 函数并获取一个日志记录器
    logger = setup_logger(__name__, logging.DEBUG)

    # 测试日志记录
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")