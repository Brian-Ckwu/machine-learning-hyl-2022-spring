import re
import sys

def clean_and_save(in_file: str, out_file: str) -> None:
    with open(in_file, "rt") as f_in:
        with open(out_file, "wt") as f_out:
            for line_in in f_in:
                line_out = re.sub(pattern=r"<unk>", repl="", string=line_in)
                line_out = re.sub(pattern=r" ", repl="", string=line_out)
                f_out.write(line_out)

if __name__ == "__main__":
    # post processing code
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    clean_and_save(in_file, out_file)