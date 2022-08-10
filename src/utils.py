"""
Author:         Victor Loveday
Date:           01/08/2022
"""

import pandas as pd


def refactor_names(names, features):
    for i, feature in enumerate(features):
        for j, name in enumerate(names):
            if name.find(f"x{i}") > -1:
                name = name.replace(f"x{i}_", f"[{feature}] ")
                name = refactor_byte_name(name)

                names[j] = name

    return names


def refactor_byte_name(name):
    name = str(name)
    name = name.replace("b'", "")
    name = name.replace("'", "")

    return name


def ravel_y(y):
    if type(y) is pd.DataFrame:
        y = y.to_numpy().ravel()

    return y


def change_label_to_class(label):
    return {
        0: "back",
        1: "buffer_overflow",
        2: "ftp_write",
        3: "guess_passwd",
        4: "imap",
        5: "ipsweep",
        6: "land",
        7: "loadmodulde",
        8: "multihop",
        9: "neptune",
        10: "nmap",
        11: "normal",
        12: "perl",
        13: "phf",
        14: "pod",
        15: "portsweep",
        16: "rootkit",
        17: "satan",
        18: "smurf",
        19: "spy",
        20: "teardrop",
        21: "warezclient",
        22: "warezmaster"
    }.get(label, "N/A")
