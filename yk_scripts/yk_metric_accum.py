import json
import os

import numpy as np
from yk_table import Table


def accumulate_metric(metrics, json_file):
    json_list = [json_file]

    acc_dict = {k: [] for k in metrics}
    for json_path in json_list:
        with open(json_path) as fp:
            data = json.load(fp)
        for k in metrics:
            if data.get(k) is not None:
                assert isinstance(data[k], (list, tuple))
                acc_dict[k].extend(data[k])

    ret = dict()
    for k in acc_dict:
        if len(acc_dict[k]) == 0:
            continue
        values = np.asarray(acc_dict[k])
        # m -> mm
        if k.startswith("mvd"):
            values *= 1000
        ret[k] = dict(mean=values.mean(), std=values.std())
    return ret


RUNS_ROOT = "runs/face_noeyeballs"
best_epochs = dict(
    m001_trump=dict(
        track=184,
        cmb3d=50,
        decmp=60,
        vocaft=161,
    ),
    # f000_watson=dict(
    #     track=60,
    #     cmb3d=60,
    #     decmp=60,
    #     vocaft=40,
    # ),
)

for speaker in ["m001_trump"]:  # , "f000_watson"]:

    metrics = ["mvd-avg:face", "mvd-avg:lips", "mvd-max:face", "mvd-max:lips"]
    ret = accumulate_metric(metrics, "yk_exp/celebtalk/m001_trump/results/metrics.json")

    def _format_values(res):
        return ["{:.3f} \u00B1{:.3f}".format(res[x]["mean"], res[x]["std"]) for x in metrics]

    table = Table("EXP", *[x + " \u00B1std" for x in metrics], alignment=("center", "middle"))
    table.add_row("voca",  *_format_values(ret))
    md = table.to_markdown()
    print(md)
    # fmt: on
