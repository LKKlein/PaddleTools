
id2type = {
    0: "bool",
    1: "int16",
    2: "int32",
    3: "int64",
    4: "float16",
    5: "float32",
    6: "float64"
}
type2id = {_type: _id for _id, _type in id2type.items()}

type2short = {
    "bool": "?",
    "int16": "h",
    "int32": "i", 
    "int64": "q",
    "float16": "",
    "float32": "f",
    "float64": "d"
}

short2size = {
    "?": 1,
    "h": 2,
    "i": 4,
    "q": 8,
    "": 2,
    "f": 4,
    "d": 8
}

email_stmp_server = {
    "qq.com": {"server": "smtp.qq.com", "port": 465, "use_ssl": True},
    "163.com": {"server": "smtp.163.com", "port": 25, "use_ssl": False},
    "gmail.com": {"server": "smtp.gmail.com", "port": 465, "use_ssl": True},
    "126.com": {"server": "smtp.126.com", "port": 25, "use_ssl": False}
}
