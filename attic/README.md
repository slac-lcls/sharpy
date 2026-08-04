# attic — retired / experimental artifacts (not the production path)

Files here are kept for reference and history but are **not** part of the active reconstruction
pipeline. Don't use them to reproduce results.

## tile_sync_test.ipynb

A 2023 prototype of **hierarchical / tiled phase synchronization**. It imports an `Operators_Tile`
module that is no longer in the tree, so it will not run as-is. It is a *separate experiment* — **not**
how the poster figures were produced.

- Production synchronization is the **power eigensolver** in `Operators.synchronize_frames_c`
  (`SHARPY_SYNC=power`, the default); `invit` is the opt-in alternative. **There is no tiled
  eigensolver in the production path.**
- The only "tiling" in production is the **KD-tree 3×3 neighbour search inside `Gramiam_plan`** — that
  builds the Gramian's overlap pairs; it is *not* an eigensolver.
- The tiled/multigrid sync idea is parked as a possible *structural* fix for the low-overlap /
  very-large-N regime (where the Fiedler cluster tightens). Revive from here if/when that work starts.

To reproduce the poster, use the production power path (see the team notes / `WORKPLAN`).
