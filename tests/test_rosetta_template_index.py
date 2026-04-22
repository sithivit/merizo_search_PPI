import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def _write_pair_list(path, pairs):
    with open(path, 'w') as f:
        for a, b in pairs:
            f.write(f"{a}:{b} X Y 1.0\n")


def _write_zhang_names(path, names):
    with open(path, 'w') as f:
        for n in names:
            f.write(n + "\n")


def test_bidirectional_index():
    from rosetta_build_template_index import build_index, load_zhang_domain_names
    with tempfile.TemporaryDirectory() as tmp:
        pair_list = os.path.join(tmp, "pairs.txt")
        names_file = os.path.join(tmp, "names.txt")
        _write_pair_list(pair_list, [("domA", "domB"), ("domC", "domD")])
        _write_zhang_names(names_file, ["domA", "domB"])

        zhang_domains = load_zhang_domain_names(names_file)
        index = build_index(pair_list, zhang_domains)

        assert "domA" in index
        assert "domB" in index["domA"]
        assert "domA" in index["domB"]  # bidirectional
        assert "domC" not in index      # not a Zhang domain
        assert "domD" not in index


def test_non_zhang_pair_excluded():
    from rosetta_build_template_index import build_index, load_zhang_domain_names
    with tempfile.TemporaryDirectory() as tmp:
        pair_list = os.path.join(tmp, "pairs.txt")
        names_file = os.path.join(tmp, "names.txt")
        _write_pair_list(pair_list, [("domX", "domY")])
        _write_zhang_names(names_file, ["domA"])  # neither X nor Y is Zhang

        zhang_domains = load_zhang_domain_names(names_file)
        index = build_index(pair_list, zhang_domains)
        assert len(index) == 0


def test_saves_and_loads_json():
    from rosetta_build_template_index import build_index, load_zhang_domain_names
    with tempfile.TemporaryDirectory() as tmp:
        pair_list = os.path.join(tmp, "pairs.txt")
        names_file = os.path.join(tmp, "names.txt")
        out_json = os.path.join(tmp, "index.json")
        _write_pair_list(pair_list, [("domA", "domB")])
        _write_zhang_names(names_file, ["domA"])

        zhang_domains = load_zhang_domain_names(names_file)
        index = build_index(pair_list, zhang_domains)

        with open(out_json, "w") as fh:
            json.dump(index, fh)
        with open(out_json) as fh:
            loaded = json.load(fh)

        assert loaded["domA"] == ["domB"]
        assert loaded["domB"] == ["domA"]
