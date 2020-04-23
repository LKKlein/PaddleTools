import logging
import os
import sys

from docopt import docopt

from .checkpoints import static2dynamic

os.umask(0)

DOCOPT_STRING = '''
Usage:
    {command} param to_dynamic --src=<source_param> --dst=<destination_param> [--verbose]

Arguments:
    param                    paddle parameters operations.

Options:
    -s <source_param>, --src=<source_param>              dir path for static params.
    -d <destination_param>, --dst=<destination_param>    where should dynamic params to store.
    -v, --verbose                                        whether to show more logs. [default: True]
'''.format(command=sys.argv[0])
cmd_args = docopt(DOCOPT_STRING, version='v1')
logger = logging.getLogger("pdtools")
logger.setLevel(logging.INFO)


def main():
    if cmd_args["param"]:
        if not os.path.exists(cmd_args["--src"]):
            raise Exception("source path: {} not exists!".format(cmd_args["--src"]))
        if cmd_args["--verbose"]:
            logger.setLevel(logging.DEBUG)
        static2dynamic(params_dir=cmd_args["--src"], save_path=cmd_args["--dst"])
