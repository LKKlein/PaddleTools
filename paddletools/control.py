import os
import sys

from docopt import docopt

from paddletools.checkpoints import static2dynamic, dynamic2static, torch2dynamic
from paddletools import logger

os.umask(0)

DOCOPT_STRING = '''
Usage:
    {command} param (to_dynamic | to_static | from_torch) --src=<source_param> \
        --dst=<destination_param> [--verbose]

Arguments:
    param                    paddle parameters operations.

Options:
    -s <source_param>, --src=<source_param>              dir path for source params.
    -d <destination_param>, --dst=<destination_param>    where should dest params to store.
    -v, --verbose                                        whether to show more logs. [default: True]

Example:
    pdtools param to_dynamic -s yolov3_pretrain/ -d yolov3 -v
'''.format(command=sys.argv[0])
cmd_args = docopt(DOCOPT_STRING, version='v1')


def main():
    if cmd_args["param"]:
        if cmd_args["--verbose"]:
            logger.setLevel("DEBUG")
        if cmd_args["to_dynamic"]:
            if not os.path.exists(cmd_args["--src"]):
                raise Exception("source path: {} not exists!".format(cmd_args["--src"]))
            static2dynamic(params_dir=cmd_args["--src"], save_path=cmd_args["--dst"])
        if cmd_args["to_static"]:
            if not os.path.exists(cmd_args["--src"] + ".pdparams"):
                raise Exception("source path: {}.pdparams not exists!".format(cmd_args["--src"]))
            dynamic2static(param_file=cmd_args["--src"], filename=cmd_args["--dst"])
        if cmd_args["from_torch"]:
            if not os.path.exists(cmd_args["--src"]):
                raise Exception("source path: {} not exists!".format(cmd_args["--src"]))
            torch2dynamic(param_file=cmd_args["--src"], save_path=cmd_args["--dst"])
