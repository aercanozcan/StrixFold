#!/usr/bin/env python3
"""golden.py pictures DIR CASE... | finalize OUT CASE... | check OUT [SUMS]: make_golden.sh's helper (stdlib)."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the published cases (SHA256SUMS, the release tarball); the others regenerate the same way
PUBLISHED = ["syn64x48", "syn700x250", "tall"]
STEPS = ["embed"] + [f"block{i:02d}" for i in range(32)] + ["norm", "rows"]
FILES = ["grid.json", "rgb.u8", "patches.bf16", "cos.f32", "sin.f32"] + [s + ".bf16" for s in STEPS] + ["span.bf16"]


def odd():
    """431x257 RGB, the engine-lab golden's odd.png pixels (a lossless PNG of exactly these)."""
    return 431, 257, bytes(v for y in range(257) for x in range(431) for v in ((x * 3 + y) % 256, (x ^ y) % 256, (y * 5 + x // 3) % 256))


def tall():
    """40x900 RGB gradient (extreme aspect ratio), the engine-lab golden's tall.png pixels."""
    return 40, 900, bytes(v for y in range(900) for x in range(40) for v in (x * 255 // 39, y * 255 // 899, (x * 7 + y // 4) % 256))


def rgba():
    """300x200 RGBA shapes, Pillow's convert("RGB") (drops alpha): needs Pillow, so it is not published."""
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (300, 200), (31, 71, 119, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((15, 20, 135, 180), fill=(240, 60, 25, 255))
    d.ellipse((110, 30, 280, 190), fill=(20, 230, 150, 112))
    d.polygon([(160, 5), (295, 70), (195, 130)], fill=(200, 40, 230, 255))
    return 300, 200, im.convert("RGB").tobytes()


CASES = {"syn64x48": "syn:64:48", "syn700x250": "syn:700:250", "odd": odd, "tall": tall, "rgba": rgba}


def pictures(folder, names):
    """Writes the raw RGB of each picture case to folder; prints every case's vision-reference argument."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        kind = CASES.get(name)
        if kind is None:
            sys.exit(f"golden.py: unknown case {name} (known: {' '.join(CASES)})")
        if isinstance(kind, str):
            print(f"{name}={kind}")
            continue
        w, h, rgb = kind()
        assert len(rgb) == w * h * 3
        (folder / f"{name}.rgb").write_bytes(rgb)
        print(f"{name}=raw:{w}:{h}:{folder.resolve() / (name + '.rgb')}")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize(out, names):
    """OUT/index.json (each case's grid, rows and span sha256) and OUT/SHA256SUMS (every case file and the index)."""
    index, lines = {}, []
    for name in sorted(names):
        folder = out / name
        present = sorted(p.name for p in folder.iterdir())
        if present != sorted(FILES):
            sys.exit(f"golden.py: {folder} holds {len(present)} files, wanted the {len(FILES)} of a case")
        grid = json.loads((folder / "grid.json").read_text())
        index[name] = {"grid": grid, "rows.bf16_sha256": sha(folder / "rows.bf16"), "span.bf16_sha256": sha(folder / "span.bf16")}
        lines += [f"{sha(folder / f)}  {name}/{f}\n" for f in present]
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    lines.append(f"{sha(out / 'index.json')}  index.json\n")
    (out / "SHA256SUMS").write_text("".join(sorted(lines, key=lambda s: s.split("  ", 1)[1])))
    print(f"{out / 'SHA256SUMS'}: {len(lines)} files")


def check(out, sums):
    """Every file SUMS lists (the published golden by default) is in OUT with that sha256."""
    bad = 0
    lines = sums.read_text().splitlines()
    for line in lines:
        digest, name = line.split("  ", 1)
        path = out / name
        got = sha(path) if path.is_file() else "missing"
        if got != digest:
            bad += 1
            print(f"DIFFERS {name}: {got}, wanted {digest}")
    print(f"{'PASS' if bad == 0 else 'FAIL'}: {len(lines) - bad} of {len(lines)} files equal to {sums}")
    return 1 if bad else 0


def main(argv):
    if len(argv) >= 3 and argv[1] == "pictures":
        pictures(Path(argv[2]), argv[3:] or PUBLISHED)
    elif len(argv) >= 3 and argv[1] == "finalize":
        finalize(Path(argv[2]), argv[3:] or PUBLISHED)
    elif len(argv) in (3, 4) and argv[1] == "check":
        return check(Path(argv[2]), Path(argv[3]) if len(argv) == 4 else HERE / "SHA256SUMS")
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
