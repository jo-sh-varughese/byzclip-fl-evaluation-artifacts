import re

tex = open("../paper/main_v2.tex", encoding="utf-8").read()
bib = open("../paper/main.bib", encoding="utf-8").read()

cite_keys = set()
for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        cite_keys.add(k.strip())
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
print("CITED but MISSING from bib:", cite_keys - bib_keys)
print("---")
labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex))
print("REF but MISSING label:", refs - labels)
print("---")
print("all cite keys used:", sorted(cite_keys))
