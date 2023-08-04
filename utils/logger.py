import logging



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # [%(filename)s][line:%(lineno)d][%(levelname)s]
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logging.root.setLevel(logging.NOTSET)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger