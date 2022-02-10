import subprocess
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, Future
import logging

"""
logging.basicConfig(
    level=logging.DEBUG,  # 控制台打印的日志级别
    filename="./outputs/code2bin.log",
    filemode="a",  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
    # a是追加模式，默认如果不写的话，就是追加模式
    format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    # 日志格式
)
"""


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    log_formatter = logging.Formatter("%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(log_formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def run_cmd(cmd: Iterable[str]):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )

    stdout, stderr = process.communicate()

    if process.returncode == 2:
        # print(stderr.decode("utf-8").rstrip())
        # return None
        msg = stderr.decode("utf-8").rstrip()
        raise ValueError(f"{' '.join(cmd)}\n{msg}")
    elif process.returncode == 9 or process.returncode == -9:
        # print(f"Program graph construction exceeded {TIMEOUT} seconds")
        # return None
        raise TimeoutError(f"{' '.join(cmd)}\ncmd timeout")
    elif process.returncode:
        # print(stderr.decode("utf-8").rstrip())
        # return None
        msg = stderr.decode("utf-8").rstrip()
        raise OSError(f"{' '.join(cmd)}\n{msg}")


def run_cmds_parallel(cmds: Iterable[list], logger:logging.Logger=None, num_workers: int = 16):
    def handle_exception(future: Future):
        exp = future.exception()
        if exp and logger is not None:
            logger.error(exp)

    with ThreadPoolExecutor(num_workers) as thread_pool:
        for cmd in cmds:
            print(cmd)
            task = thread_pool.submit(run_cmd, cmd)
            task.add_done_callback(handle_exception)


if __name__ == '__main__':        
    # first file logger
    logger = setup_logger('first_logger', 'first_logfile.log')
    logger.info('This is just info message')

    # second file logger
    super_logger = setup_logger('second_logger', 'second_logfile.log')
    super_logger.error('This is an error message')