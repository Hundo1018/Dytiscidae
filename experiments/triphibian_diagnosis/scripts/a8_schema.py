"""A8 schema dump: nested keys with types of JSON lines / JSON files.

Streams; never loads a whole .jsonl.  Usage:
  python a8_schema.py line <file.jsonl> first|last          # one line
  python a8_schema.py events <events.jsonl> [max_types]     # first line of each event type
  python a8_schema.py archive <archive.json> [n_elites]     # top-level + first elites
"""
import json
import os
import sys


def shape(v, depth=0, maxdepth=6, maxkeys=80):
    pad = "  " * depth
    out = []
    if isinstance(v, dict):
        keys = list(v.keys())
        for k in keys[:maxkeys]:
            x = v[k]
            t = type(x).__name__
            if isinstance(x, dict):
                out.append(f"{pad}{k}: dict[{len(x)}]")
                if depth < maxdepth:
                    out.extend(shape(x, depth + 1, maxdepth, maxkeys))
            elif isinstance(x, list):
                inner = type(x[0]).__name__ if x else "?"
                sample = ""
                if x and not isinstance(x[0], (dict, list)):
                    sample = " e.g. " + json.dumps(x[:4])[:80]
                out.append(f"{pad}{k}: list[{len(x)}] of {inner}{sample}")
                if x and isinstance(x[0], dict) and depth < maxdepth:
                    out.extend(shape(x[0], depth + 1, maxdepth, maxkeys))
            else:
                out.append(f"{pad}{k}: {t} = {json.dumps(x)[:70]}")
        if len(keys) > maxkeys:
            out.append(f"{pad}... {len(keys) - maxkeys} more keys")
    return out


def last_line(path):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        pos = end - 1
        buf = b""
        while pos >= 0:
            step = min(65536, pos + 1)
            pos -= step
            f.seek(pos + 1)
            buf = f.read(step) + buf
            s = buf.rstrip(b"\n")
            i = s.rfind(b"\n")
            if i >= 0:
                return s[i + 1:].decode()
        return buf.decode().strip()


def main():
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "line":
        which = sys.argv[3]
        if which == "first":
            with open(path) as f:
                line = f.readline()
        else:
            line = last_line(path)
        print("\n".join(shape(json.loads(line))))
    elif mode == "events":
        maxt = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        seen = {}
        counts = {}
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("type") or r.get("event") or r.get("kind") or "?"
                counts[t] = counts.get(t, 0) + 1
                if t not in seen and len(seen) < maxt:
                    seen[t] = r
        print("event type counts:", json.dumps(counts))
        for t, r in seen.items():
            print(f"===== {t}")
            print("\n".join(shape(r, maxdepth=3, maxkeys=60)))
    elif mode == "keyunion":
        # every key path (depth <= maxd) over all lines, with line count and first generation seen
        maxd = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        cnt, first, types = {}, {}, {}
        nlines = 0

        def walk(v, pre, d, g):
            if not isinstance(v, dict) or d > maxd:
                return
            for k, x in v.items():
                p = pre + "." + k if pre else k
                cnt[p] = cnt.get(p, 0) + 1
                types.setdefault(p, set()).add(type(x).__name__)
                if p not in first:
                    first[p] = g
                if isinstance(x, dict) and len(x) < 60:
                    walk(x, p, d + 1, g)
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nlines += 1
                g = r.get("generation", r.get("gen", r.get("kind", "?")))
                walk(r, "", 1, g)
        print("lines", nlines)
        for p in sorted(cnt):
            print(f"{p}\tn={cnt[p]}\tfirst={first[p]}\t{'/'.join(sorted(types[p]))}")
    elif mode == "archive":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        d = json.load(open(path))
        top = {k: (v if not isinstance(v, list) else v[:0]) for k, v in d.items()} if isinstance(d, dict) else {}
        print("top-level:", {k: type(v).__name__ + (f"[{len(v)}]" if isinstance(v, (list, dict)) else "") for k, v in d.items()})
        print("\n".join(shape({k: v for k, v in d.items() if not isinstance(v, list)}, maxdepth=2)))
        el = d.get("elites") or []
        for e in el[:n]:
            print("===== elite")
            print("\n".join(shape(e, maxdepth=4, maxkeys=120)))


if __name__ == "__main__":
    main()
