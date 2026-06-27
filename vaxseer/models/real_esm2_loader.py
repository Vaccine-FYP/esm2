"""
real_esm2_loader.py  —  place in vaxseer/models/

Loads the REAL fair-esm (for ESM-2) even though the repo vendors an OLD `esm/`
fork that shadows it. The real package uses absolute imports (`from esm.x import
y`) internally, so aliasing the top module is not enough — during the real
import, `sys.modules["esm"]` must point at the real package. We therefore:

  1. stash the vendored `esm` (and its submodules) out of sys.modules
  2. put site-packages first on sys.path
  3. import the real esm + the ESM-2 builder, build the model+alphabet
  4. restore the vendored `esm` so the rest of VaxSeer (MSA model, alphabets) works

This swap happens only at model-construction time (once), so it does not affect
the vendored esm used elsewhere.

Install once:  pip install fair-esm==2.0.0
"""
import os
import sys
import importlib


def _site_packages_dirs():
    return [p for p in sys.path if "site-packages" in p or "dist-packages" in p]


def load_esm2(model_name="esm2_t30_150M_UR50D"):
    """Return (model, alphabet) for the requested ESM-2 size using the real
    fair-esm, without permanently disturbing the vendored `esm` package."""

    # --- 1. stash any already-imported esm modules (the vendored fork) ---
    stashed = {}
    for name in list(sys.modules.keys()):
        if name == "esm" or name.startswith("esm."):
            stashed[name] = sys.modules.pop(name)

    # --- 2. ensure the real fair-esm in site-packages is importable first ---
    site_dirs = _site_packages_dirs()
    real_root = None
    for d in site_dirs:
        if os.path.isfile(os.path.join(d, "esm", "rotary_embedding.py")):
            real_root = d
            break
    if real_root is None:
        sys.modules.update(stashed)
        raise ImportError(
            "Real fair-esm (with esm/rotary_embedding.py) not found in "
            "site-packages. Install it: pip install fair-esm==2.0.0"
        )

    original_path = list(sys.path)
    try:
        sys.path.insert(0, real_root)
        importlib.invalidate_caches()

        real_esm = importlib.import_module("esm")
        import esm.pretrained as real_pretrained  # noqa
        loader = getattr(real_pretrained, model_name)
        model, alphabet = loader()
    finally:
        for name in list(sys.modules.keys()):
            if name == "esm" or name.startswith("esm."):
                del sys.modules[name]
        sys.modules.update(stashed)
        sys.path[:] = original_path
        importlib.invalidate_caches()

    return model, alphabet


if __name__ == "__main__":
    try:
        import esm as _vendored  # noqa
        print("vendored esm imported first (mimicking bin.train):",
              getattr(_vendored, "__file__", "?"))
    except Exception as e:
        print("vendored esm import note:", e)

    print("Loading real ESM-2 via swap ...")
    model, alphabet = load_esm2("esm2_t30_150M_UR50D")
    print("Loaded:", type(model).__name__, "| embed_dim:", model.embed_dim,
          "| layers:", model.num_layers)

    import esm as _check
    print("after load, esm points to:", getattr(_check, "__file__", "?"))

    bc = alphabet.get_batch_converter()
    data = [("p1", "MKTAYIAKQR"), ("p2", "GGGSEKVLAA")]
    _, _, toks = bc(data)
    import torch
    with torch.no_grad():
        out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
    rep = out["representations"][model.num_layers][:, 0]
    print("Forward OK. BOS rep shape:", tuple(rep.shape))
    print("SUCCESS: real ESM-2 works and vendored esm restored.")
