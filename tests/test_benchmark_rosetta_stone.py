import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def test_score_pair_bridge_found():
    """P1 has A-like domain, P2 has B-like domain — bridge via template (A,B)."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9, "domX": 0.7},
        "P2": {"domB": 0.8, "domY": 0.6},
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    assert abs(score - 0.8) < 1e-6  # min(0.9, 0.8) = 0.8


def test_score_pair_no_bridge():
    """No template pair bridges P1 hits to P2 hits."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9},
        "P2": {"domC": 0.8},  # domC not paired with domA
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    assert score == 0.0


def test_score_pair_reverse_bridge():
    """Template index is bidirectional: B in H[P1], A in H[P2] also works."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domB": 0.75},
        "P2": {"domA": 0.85},
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    assert abs(score - 0.75) < 1e-6  # min(0.85, 0.75) = 0.75


def test_score_pair_best_bridge_wins():
    """With multiple bridging templates, best score is returned."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9, "domC": 0.6},
        "P2": {"domB": 0.5, "domD": 0.95},
    }
    template_index = {
        "domA": ["domB"], "domB": ["domA"],
        "domC": ["domD"], "domD": ["domC"],
    }
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    # Bridge 1: min(0.9, 0.5) = 0.5 ; Bridge 2: min(0.6, 0.95) = 0.6
    assert abs(score - 0.6) < 1e-6


def test_build_protein_hit_map_max_per_template():
    """Multiple domains in P1 — max TM per template domain is taken."""
    from benchmark_rosetta_stone import build_protein_hit_map
    per_domain_hits = {
        ("P1", "dom00"): {"tmplX": 0.7, "tmplY": 0.5},
        ("P1", "dom01"): {"tmplX": 0.9, "tmplZ": 0.4},
    }
    hit_map = build_protein_hit_map(per_domain_hits)
    assert hit_map["P1"]["tmplX"] == 0.9  # max over domains
    assert hit_map["P1"]["tmplY"] == 0.5
    assert hit_map["P1"]["tmplZ"] == 0.4


def test_score_missing_protein_returns_zero():
    """If a protein has no search results, score is 0."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {"P1": {"domA": 0.9}}
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "MISSING", domain_hits, template_index)
    assert score == 0.0


# --- Self-interaction filter tests ---

def test_self_filter_p1_has_both_domains():
    """P1 already carries both A-like and B-like domains: bridge is trivially
    explained by P1's intrachain contact and must be rejected."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9, "domB": 0.8},  # P1 has both sides of the bridge
        "P2": {"domB": 0.8},
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    assert score == 0.0


def test_self_filter_p2_has_both_domains():
    """P2 already carries both A-like and B-like domains: bridge rejected."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9},
        "P2": {"domA": 0.7, "domB": 0.8},  # P2 has both sides of the bridge
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    assert score == 0.0


def test_self_filter_valid_when_domains_split():
    """Valid Rosetta Stone: P1 has only A-like, P2 has only B-like.
    The bridge is genuinely interchain and must survive the filter."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9},
        "P2": {"domB": 0.8},
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index)
    assert abs(score - 0.8) < 1e-6  # min(0.9, 0.8) = 0.8


def test_self_filter_threshold_respected():
    """With self_filter_tm=0.5, a weak B hit in P1 (TM=0.3) should NOT
    trigger the filter — the bridge remains valid."""
    from benchmark_rosetta_stone import score_pair_rosetta
    domain_hits = {
        "P1": {"domA": 0.9, "domB": 0.3},  # domB present but below self_filter_tm
        "P2": {"domB": 0.8},
    }
    template_index = {"domA": ["domB"], "domB": ["domA"]}
    score = score_pair_rosetta("P1", "P2", domain_hits, template_index,
                               min_bridge_tm=0.0, self_filter_tm=0.5)
    assert abs(score - 0.8) < 1e-6  # bridge valid: P1's domB hit is below 0.5
