# topology.py

NODES = ["h1", "s1", "s2", "s3", "s4", "h2"]

LINKS = [
    ("h1", "s1"),
    ("h1", "s2"),
    ("h1", "s3"),
    ("s1", "s4"),
    ("s2", "s4"),
    ("s3", "s4"),
    ("s4", "h2"),
]

LINK_INDEX = {link: i for i, link in enumerate(LINKS)}
NUM_LINKS = len(LINKS)
