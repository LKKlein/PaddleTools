from .checkpoints import static2dynamic
from docopt import docopt
import sys
import os

os.umask(0)

DOCOPT_STRING = '''
Usage:
    {command} param to_dynamic --src=<source_param> --dst=<destination_param>

Arguments:
    param                    paddle parameters operations.

Options:
    -s <source_param>, --src=<source_param>              dir path for static params.
    -d <destination_param>, --dst=<destination_param>    where should dynamic params to store.
'''.format(command=sys.argv[0])
cmd_args = docopt(DOCOPT_STRING, version='v1')


def main():
    if cmd_args["param"]:
        if not os.path.exists(cmd_args["--src"]):
            raise Exception("source path: {} not exists!".format(cmd_args["--src"]))
        static2dynamic(params_dir=cmd_args["--src"], save_path=cmd_args["--dst"])
        