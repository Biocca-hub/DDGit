"""
Unisce a_pos_features.pt (N x 77) e b_surr_features.pt (N x 112) in
../output/aug_features.pt (N x 127):

    colonne   0-61   -> feature comuni baseline (identiche in a e b)
    colonne  62-76   -> feature specifiche di a ovvero delta chimico fisici (a[:, 62:77], 15 colonne)
    colonne  77-126  -> feature specifiche di b ovvero medie proprieta cf intorni strutturali diversi raggi (b[:, 62:112], 50 colonne)

(indici 0-based; in 1-based: 1-62 comuni, 63-77 da a, 78-127 da b)
"""
from pathlib import Path

import torch

INPUT_DIR = Path("../input")
OUTPUT_PATH = Path("../output/aug_features.pt")

N_COMMON = 62   # feature condivise
A_COLS = 77
B_COLS = 112


def main() -> None:
    a = torch.load(INPUT_DIR / "a_pos_features.pt", weights_only=True)
    b = torch.load(INPUT_DIR / "b_surr_features.pt", weights_only=True)

    # 1. controllo shape
    assert a.dim() == 2 and a.shape[1] == A_COLS, f"a ha shape {tuple(a.shape)}, attesa (N, {A_COLS})"
    assert b.dim() == 2 and b.shape[1] == B_COLS, f"b ha shape {tuple(b.shape)}, attesa (N, {B_COLS})"
    assert a.shape[0] == b.shape[0], f"numero di righe diverso: a={a.shape[0]}, b={b.shape[0]}"
    n = a.shape[0]
    print(f"a: {tuple(a.shape)}  b: {tuple(b.shape)}  N = {n}")

    # 2. le prime 62 feature devono coincidere riga per riga
    row_ok = (a[:, :N_COMMON] == b[:, :N_COMMON]).all(dim=1)
    if not row_ok.all():
        bad = (~row_ok).nonzero(as_tuple=True)[0]
        raise ValueError(
            f"{len(bad)} righe con feature comuni diverse tra a e b "
            f"(prime: {bad[:10].tolist()})"
        )
    print(f"OK: le prime {N_COMMON} feature coincidono in tutte le {n} righe")

    # 3. costruzione del nuovo tensore
    common = a[:, :N_COMMON]
    a_spec = a[:, N_COMMON:A_COLS]
    b_spec = b[:, N_COMMON:B_COLS]
    aug = torch.cat([common, a_spec, b_spec], dim=1)

    # verifica che ogni blocco sia stato copiato correttamente, riga per riga
    end_a = N_COMMON + a_spec.shape[1]
    checks = {
        "comuni   (aug[:, 0:62]   == a[:, 0:62])": aug[:, :N_COMMON] == a[:, :N_COMMON],
        "da a     (aug[:, 62:77]  == a[:, 62:77])": aug[:, N_COMMON:end_a] == a[:, N_COMMON:A_COLS],
        "da b     (aug[:, 77:127] == b[:, 62:112])": aug[:, end_a:] == b[:, N_COMMON:B_COLS],
    }
    for name, eq in checks.items():
        bad = (~eq.all(dim=1)).nonzero(as_tuple=True)[0]
        if len(bad):
            raise ValueError(f"blocco {name}: {len(bad)} righe non corrispondono (prime: {bad[:10].tolist()})")
        print(f"OK: {name}")

    expected_cols = N_COMMON + (A_COLS - N_COMMON) + (B_COLS - N_COMMON)
    assert aug.shape == (n, expected_cols), f"shape finale {tuple(aug.shape)}, attesa ({n}, {expected_cols})"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aug, OUTPUT_PATH)
    print(f"Salvato {OUTPUT_PATH}  shape {tuple(aug.shape)}  dtype {aug.dtype}")


if __name__ == "__main__":
    main()