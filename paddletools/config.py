
id2type = {
    0: "bool",
    1: "int16",
    2: "int32",
    3: "int64",
    4: "float16",
    5: "float32",
    6: "float64"
}

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
