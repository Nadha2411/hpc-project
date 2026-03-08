#ifndef DATA_LOADER_H
#define DATA_LOADER_H

/* Binary dataset loader for CB513 preprocessed splits.
 *
 * Reads flat binary files produced by scripts/preprocess/cb513_prepare.py:
 *   <binary_dir>/<split>_X.bin  — float32, row-major [n x feature_dim]
 *   <binary_dir>/<split>_y.bin  — int32,   [n]
 *   <binary_dir>/binary_meta.json — shapes and metadata
 *
 * C loading contract (no in-file headers):
 *   fread(ds->X, sizeof(float), ds->n * ds->feature_dim, fx);
 *   fread(ds->y, sizeof(int),   ds->n,                   fy);
 *
 * Usage:
 *   Dataset train, val, test;
 *   dataset_load(&train, "data/processed/cb513/binary", "train");
 *   dataset_load(&val,   "data/processed/cb513/binary", "val");
 *   dataset_load(&test,  "data/processed/cb513/binary", "test");
 *   // ... train ...
 *   dataset_free(&train); dataset_free(&val); dataset_free(&test);
 */

typedef struct {
    float *X;        /* [n x feature_dim] row-major, float32 */
    int   *y;        /* [n] class ids: 0=H, 1=E, 2=C */
    int    n;        /* number of samples */
    int    feature_dim; /* 260 (20 AA x 13 window) */
} Dataset;

/* Load one split ("train", "val", or "test") from binary_dir.
 * Reads n_samples from binary_meta.json, then fread-loads X and y.
 * Exits with error message on any failure. */
void dataset_load(Dataset *ds, const char *binary_dir, const char *split);

/* Free memory allocated by dataset_load(). */
void dataset_free(Dataset *ds);

#endif /* DATA_LOADER_H */
