import sys
import re
import pymorphy2 as pm2
# import module that will extract recommends


def main(msg: str):
    morphy = pm2.MorphAnalyzer()

    norm_msg_pos = "_".join([morphy.normal_forms(x)[0] for x in re.findall(r"\w+", msg)])

    return norm_msg_pos


if __name__ == "__main__":
    main(sys.argv[1])
