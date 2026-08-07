import sys
import zipfile
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def para_text(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def dump(path):
    print("=" * 78)
    print(path)
    print("=" * 78)
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    body = root.find(W + "body")
    for el in body:
        tag = el.tag
        if tag == W + "p":
            t = para_text(el).strip()
            if t:
                print(t)
        elif tag == W + "tbl":
            print("\n--- TABLE ---")
            for tr in el.findall(W + "tr"):
                cells = []
                for tc in tr.findall(W + "tc"):
                    txt = " ".join(
                        para_text(p).strip() for p in tc.findall(W + "p")
                    ).strip()
                    cells.append(txt)
                print(" | ".join(cells))
            print("--- END TABLE ---\n")


for p in sys.argv[1:]:
    dump(p)
