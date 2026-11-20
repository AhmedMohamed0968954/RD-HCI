"""Member 2's frozen STA probe applied once to every clip, as memory steps for Member 3's AGA.

The memory used to summarise each past second by the mean of its 24 V-JEPA tokens, while the present
clip goes through Member 2's FactorizedAttentiveProbe. Because that probe is frozen, its outputs for
every bank clip can be computed once and stored in the consolidated layout (bank.py), with one
"token" per clip:

    <out_dir>/bank_tokens.npy     [M, 1, 2E + 97 + 300] float16, same rows as the source bank
    <out_dir>/sta_obs_tokens.npy  [N, 1, 2E + 97 + 300] float16, same rows as the source STA clips
    <out_dir>/bank_index.csv, events_index.csv   copied from the source (same times, same rows)
    <out_dir>/meta.json           source features, probe checkpoint, layout (written last)

Each row = [verb embedding e_v (E) | noun embedding e_n (E) | verb logits (97) | noun logits (300)],
exactly `probe.encode_tokens` + `verb_head` / `noun_head` as the present branch uses them. FeatureBank
and Anchors read the folder unchanged; AGAMemoryModel(step_encoder="probe") splits the row with
`layout()`. No timing changes: every row comes from the same clip as the source row.
"""
from __future__ import annotations
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch

N_VERB, N_NOUN = 97, 300
ARRAYS = ("bank_tokens.npy", "sta_obs_tokens.npy")
INDEXES = ("bank_index.csv", "events_index.csv")


def layout(embed_dim: int) -> dict[str, tuple[int, int]]:
    """Column ranges of one stored row."""
    E = embed_dim
    return {"verb_emb": (0, E), "noun_emb": (E, 2 * E), "verb_logits": (2 * E, 2 * E + N_VERB),
            "noun_logits": (2 * E + N_VERB, 2 * E + N_VERB + N_NOUN)}


def step_dim(embed_dim: int) -> int:
    return 2 * embed_dim + N_VERB + N_NOUN


def file_id(path) -> dict:
    """Identity of a checkpoint or array file (path, size, modification time)."""
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def same_file(recorded: dict | None, path) -> bool:
    if not recorded:
        return False
    now = file_id(path)
    return recorded.get("size") == now["size"] and recorded.get("mtime_ns") == now["mtime_ns"]


def load_probe(path, embed_dim: int):
    """Member 2's STA factorized probe checkpoint (train_probe.py best.pt), in eval mode."""
    from epic_starter.models.factorized_probe import FactorizedAttentiveProbe
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("mode") not in (None, "factorized") or ckpt.get("task") not in (None, "STA"):
        raise ValueError(f"Expected an STA 'factorized' probe checkpoint, got task={ckpt.get('task')} "
                         f"mode={ckpt.get('mode')}")
    probe = FactorizedAttentiveProbe(embed_dim=embed_dim, mode="factorized")
    probe.load_state_dict(ckpt["model"])
    return probe.eval().requires_grad_(False)


@torch.no_grad()
def encode(probe, tokens: torch.Tensor) -> torch.Tensor:
    """[B, 24, E] -> [B, 2E + 397]: embeddings and logits, as the present branch computes them."""
    ev, en = probe.encode_tokens(tokens.float())
    return torch.cat([ev, en, probe.verb_head(ev), probe.noun_head(en)], dim=-1)


def precompute(features_dir, probe_ckpt, out_dir, batch_size: int = 512, log=print) -> Path:
    """Write <out_dir> as described above. Re-running with the same source and probe is a no-op;
    a different probe or source in an existing <out_dir> is refused (use a new folder)."""
    src, out = Path(features_dir), Path(out_dir)
    src_meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").is_file() else {}
    probe_id, bank_id = file_id(probe_ckpt), file_id(src / "bank_tokens.npy")
    if (out / "meta.json").is_file():
        meta = json.loads((out / "meta.json").read_text())
        if same_file(meta.get("probe_checkpoint"), probe_ckpt) and same_file(meta.get("source_bank"),
                                                                              src / "bank_tokens.npy"):
            log(f"already done: {out}")
            return out
        raise FileExistsError(f"{out} holds probe steps of another probe or source; choose a new --out-dir")
    out.mkdir(parents=True, exist_ok=True)
    embed_dim = int(np.load(src / "bank_tokens.npy", mmap_mode="r").shape[2])
    probe = load_probe(probe_ckpt, embed_dim)
    dim = step_dim(embed_dim)
    for name in ARRAYS:
        arr = np.load(src / name, mmap_mode="r")
        if arr.ndim != 3 or arr.shape[1] != 24 or arr.shape[2] != embed_dim:
            raise ValueError(f"{src / name}: expected [N, 24, {embed_dim}], got {arr.shape}")
        tmp = out / f"{Path(name).stem}.{os.getpid()}.tmp.npy"
        dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(arr.shape[0], 1, dim))
        t0, n = time.time(), arr.shape[0]
        for i in range(0, n, batch_size):
            x = torch.from_numpy(np.asarray(arr[i:i + batch_size], dtype=np.float32))
            dst[i:i + batch_size, 0] = encode(probe, x).numpy().astype(np.float16)
            done = min(i + batch_size, n)
            if done == n or (i // batch_size) % 100 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                log(f"{name}: {done}/{n} clips ({rate:.0f}/s, ~{(n - done) / max(rate, 1e-9) / 60:.1f} min left)")
        dst.flush()
        del dst
        os.replace(tmp, out / name)
    for name in INDEXES:
        shutil.copyfile(src / name, out / name)
    meta = {"step_features": "member2_sta_probe", "source_features": str(src.resolve()),
            "source_bank": bank_id, "source_meta": src_meta, "probe_checkpoint": probe_id,
            "embed_dim": embed_dim, "dim": dim, "tokens": 1, "dtype": "float16",
            "layout": layout(embed_dim), "bank_stride_s": src_meta.get("bank_stride_s"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))     # last: marks the folder complete
    return out
