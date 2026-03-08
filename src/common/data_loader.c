#include "common/data_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Minimal JSON field extractor.
 * Searches for "\"<key>\": <integer>" in the buffer and returns the value.
 * Returns -1 if not found. Only handles integer values (sufficient for
 * reading n_samples and feature_dim from binary_meta.json).
 * ----------------------------------------------------------------------- */
static int json_find_int(const char *buf, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(buf, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p);
}

/* Search for n_samples under the split section.
 * Locates the split name block and reads n_samples from within it. */
static int json_find_split_n(const char *buf, const char *split) {
    /* Find the split key, e.g. "\"train\":" */
    char key[64];
    snprintf(key, sizeof(key), "\"%s\":", split);
    const char *p = strstr(buf, key);
    if (!p) return -1;
    p += strlen(key);
    /* Find n_samples within the next 512 bytes (the split object) */
    char block[512];
    strncpy(block, p, sizeof(block) - 1);
    block[sizeof(block) - 1] = '\0';
    return json_find_int(block, "n_samples");
}

void dataset_load(Dataset *ds, const char *binary_dir, const char *split) {
    char meta_path[512], x_path[512], y_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/binary_meta.json", binary_dir);
    snprintf(x_path,    sizeof(x_path),    "%s/%s_X.bin",         binary_dir, split);
    snprintf(y_path,    sizeof(y_path),    "%s/%s_y.bin",         binary_dir, split);

    /* --- Read binary_meta.json --- */
    FILE *fmeta = fopen(meta_path, "r");
    if (!fmeta) {
        fprintf(stderr, "[data_loader] Cannot open %s\n", meta_path);
        exit(1);
    }
    fseek(fmeta, 0, SEEK_END);
    long meta_size = ftell(fmeta);
    rewind(fmeta);
    char *meta_buf = (char *)malloc((size_t)meta_size + 1);
    if (!meta_buf) { fprintf(stderr, "[data_loader] OOM reading meta\n"); exit(1); }
    size_t meta_read = fread(meta_buf, 1, (size_t)meta_size, fmeta);
    (void)meta_read; /* size already known from fseek */
    meta_buf[meta_size] = '\0';
    fclose(fmeta);

    int feature_dim = json_find_int(meta_buf, "feature_dim");
    int n_samples   = json_find_split_n(meta_buf, split);
    free(meta_buf);

    if (feature_dim <= 0) {
        fprintf(stderr, "[data_loader] Could not parse feature_dim from %s\n", meta_path);
        exit(1);
    }
    if (n_samples <= 0) {
        fprintf(stderr, "[data_loader] Could not parse n_samples for split '%s'\n", split);
        exit(1);
    }

    ds->n           = n_samples;
    ds->feature_dim = feature_dim;

    /* --- Allocate --- */
    ds->X = (float *)malloc((size_t)n_samples * (size_t)feature_dim * sizeof(float));
    ds->y = (int   *)malloc((size_t)n_samples * sizeof(int));
    if (!ds->X || !ds->y) {
        fprintf(stderr, "[data_loader] OOM allocating dataset split '%s'\n", split);
        exit(1);
    }

    /* --- Load X --- */
    FILE *fx = fopen(x_path, "rb");
    if (!fx) {
        fprintf(stderr, "[data_loader] Cannot open %s\n", x_path);
        exit(1);
    }
    size_t x_read = fread(ds->X, sizeof(float), (size_t)n_samples * (size_t)feature_dim, fx);
    fclose(fx);
    if ((int)x_read != n_samples * feature_dim) {
        fprintf(stderr, "[data_loader] X read %zu items, expected %d\n",
                x_read, n_samples * feature_dim);
        exit(1);
    }

    /* --- Load y --- */
    FILE *fy = fopen(y_path, "rb");
    if (!fy) {
        fprintf(stderr, "[data_loader] Cannot open %s\n", y_path);
        exit(1);
    }
    size_t y_read = fread(ds->y, sizeof(int), (size_t)n_samples, fy);
    fclose(fy);
    if ((int)y_read != n_samples) {
        fprintf(stderr, "[data_loader] y read %zu items, expected %d\n", y_read, n_samples);
        exit(1);
    }
}

void dataset_free(Dataset *ds) {
    free(ds->X); ds->X = NULL;
    free(ds->y); ds->y = NULL;
    ds->n = 0;
    ds->feature_dim = 0;
}
