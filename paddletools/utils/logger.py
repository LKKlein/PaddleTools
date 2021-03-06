# coding:utf-8
import logging
import math
import os

import colorlog


class Logger(object):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    TRAIN = 21
    EVAL = 22
    PLACEHOLDER = '%'
    NOLOG = "NOLOG"
    logging.addLevelName(TRAIN, 'TRAIN')
    logging.addLevelName(EVAL, 'EVAL')

    def __init__(self, name="PaddleTools", level="INFO"):
        self.logger = logging.getLogger(name)
        self.handlers = []
        self.log_colors = {
            'DEBUG': 'purple',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
            'TRAIN': 'cyan',
            'EVAL': 'blue',
        }
        self.handler = logging.StreamHandler()

        self.format = colorlog.ColoredFormatter(
            '%(log_color)s[%(asctime)-15s] [%(levelname)8s] - %(message)s',
            log_colors=self.log_colors)
        self.handler.setFormatter(self.format)
        self.logger.addHandler(self.handler)
        self.handlers.append(self.handler)
        self.logLevel = level
        assert hasattr(logging, level), "logging has no level named {}".format(level)
        self.logger.setLevel(self.logLevel)
        self.logger.propagate = False

    def log_to_file(self, filename, including_all=True):
        """
        Saving logs to specific file.

        :Parameters:
        - filename (str): use which file to save logs. If path don't exist, it will auto create.
        - including_all (bool): whether to save all other logger's logs to file.
        """
        parent_dir = os.path.realpath(os.path.dirname(filename))
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        filehandler = logging.FileHandler(filename)
        file_format = logging.Formatter('[%(asctime)-15s] [%(levelname)8s] - %(message)s')
        filehandler.setFormatter(file_format)
        self.handlers.append(filehandler)

        if including_all:
            for logger_name, logger in logging.root.manager.loggerDict.items():
                if isinstance(logger, logging.Logger):
                    logger.addHandler(filehandler)
        else:
            self.logger.addHandler(filehandler)

    def set_format(self, formats):
        color_format = colorlog.ColoredFormatter(
            formats, log_colors=self.log_colors
        )
        for handler in self.handlers:
            handler.setFormatter(color_format)

    def _is_no_log(self):
        return self.getLevel() == Logger.NOLOG

    def setLevel(self, logLevel):
        assert logLevel in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.logLevel = logLevel.upper()
        if not self._is_no_log():
            _logging_level = eval("logging.%s" % self.logLevel)
            self.logger.setLevel(_logging_level)

    def getLevel(self):
        return self.logLevel

    def __call__(self, level, msg):
        def _get_log_arr(msg, len_limit=150):
            ph = Logger.PLACEHOLDER
            lrspace = 2
            lc = rc = " " * lrspace
            tbspace = 1
            msgarr = str(msg).split("\n")
            if len(msgarr) == 1:
                return msgarr

            temp_arr = msgarr
            msgarr = []
            for text in temp_arr:
                if len(text) > len_limit:
                    for i in range(math.ceil(len(text) / len_limit)):
                        if i == 0:
                            msgarr.append(text[0:len_limit])
                        else:
                            fr = len_limit + (len_limit - 4) * (i - 1)
                            to = len_limit + (len_limit - 4) * i
                            if to > len(text):
                                to = len(text)
                            msgarr.append("===>" + text[fr:to])
                else:
                    msgarr.append(text)

            maxlen = -1
            for text in msgarr:
                if len(text) > maxlen:
                    maxlen = len(text)

            result = [" ", ph * (maxlen + 2 + lrspace * 2)]
            tbline = "%s%s%s" % (ph, " " * (maxlen + lrspace * 2), ph)
            for index in range(tbspace):
                result.append(tbline)
            for text in msgarr:
                text = "%s%s%s%s%s%s" % (ph, lc, text, rc, " " *
                                         (maxlen - len(text)), ph)
                result.append(text)
            for index in range(tbspace):
                result.append(tbline)
            result.append(ph * (maxlen + 2 + lrspace * 2))
            return result

        if self._is_no_log():
            return

        for msg in _get_log_arr(msg):
            self.logger.log(level, msg)

    def debug(self, msg):
        self(logger.DEBUG, msg)

    def info(self, msg):
        self(logger.INFO, msg)

    def warning(self, msg):
        self(logger.WARNING, msg)

    def error(self, msg):
        self(logger.ERROR, msg)

    def critical(self, msg):
        self(logger.CRITICAL, msg)

    def train(self, msg):
        self(logger.TRAIN, msg)

    def eval(self, msg):
        self(logger.EVAL, msg)


logger = Logger()
