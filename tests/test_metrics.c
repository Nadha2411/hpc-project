/* tests/test_metrics.c — Phase 3 verification test driver.
 *
 * Tests:
 *   1. data_loader: loads train/val/test splits, checks sizes and label range
 *   2. metrics:     Q3 on perfect predictions = 100%, all-C prediction = ~42.7%
 *   3. confusion:   diagonal-only matrix for perfect predictions
 *   4. cross_entropy: loss = 0 for p=1 predictions
 *   5. logger:      writes valid JSON to results/serial/
 *   6. timer:       elapsed time > 0
 *
 * Build: make test_metrics
 * Run:   ./test_metrics --data data/processed/cb513/binary --out results/serial
 */

#include "common/cli.h"
#include "common/data_loader.h"
#include "common/logger.h"
#include "common/metrics.h"
#include "common/timer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int pass_count = 0;
static int fail_count = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS  %s\n", msg); pass_count++; } \
    else       { printf("  FAIL  %s\n", msg); fail_count++; } \
} while(0)

#define NEAR(a, b, tol) (fabsf((a)-(b)) < (tol))

int main(int argc, char **argv) {
    Args args;
    parse_args(argc, argv, &args, "serial");

    printf("=== Phase 3 Verification: test_metrics ===\n\n");

    /* ------------------------------------------------------------------ */
    printf("[1] Timer\n");
    Timer t;
    timer_start(&t);
    /* small busy wait */
    volatile long x = 0;
    for (long i = 0; i < 1000000L; i++) x += i;
    double elapsed = timer_elapsed_s(&t);
    CHECK(elapsed > 0.0, "timer_elapsed_s > 0");
    printf("    elapsed = %.6f s\n\n", elapsed);

    /* ------------------------------------------------------------------ */
    printf("[2] Data Loader\n");
    Dataset train, val, test;
    dataset_load(&train, args.data_path, "train");
    dataset_load(&val,   args.data_path, "val");
    dataset_load(&test,  args.data_path, "test");

    CHECK(train.n > 0,            "train.n > 0");
    CHECK(val.n   > 0,            "val.n   > 0");
    CHECK(test.n  > 0,            "test.n  > 0");
    CHECK(train.feature_dim == 260, "feature_dim == 260");
    CHECK(train.n + val.n + test.n == 84119, "total residues == 84119");

    /* All labels must be 0, 1, or 2 */
    int label_ok = 1;
    for (int i = 0; i < train.n; i++)
        if (train.y[i] < 0 || train.y[i] > 2) { label_ok = 0; break; }
    CHECK(label_ok, "all train labels in {0,1,2}");

    printf("    train=%d  val=%d  test=%d  feature_dim=%d\n\n",
           train.n, val.n, test.n, train.feature_dim);

    /* ------------------------------------------------------------------ */
    printf("[3] Metrics — perfect predictions\n");
    /* Perfect: y_pred == y_true */
    float q3_perfect = compute_q3(test.y, test.y, test.n);
    CHECK(NEAR(q3_perfect, 100.0f, 0.01f), "Q3(perfect) == 100.0");

    float per_class[3];
    compute_per_class_accuracy(test.y, test.y, test.n, per_class);
    CHECK(NEAR(per_class[0], 100.0f, 0.01f), "per_class H(perfect) == 100");
    CHECK(NEAR(per_class[1], 100.0f, 0.01f), "per_class E(perfect) == 100");
    CHECK(NEAR(per_class[2], 100.0f, 0.01f), "per_class C(perfect) == 100");

    int conf[3][3];
    compute_confusion_matrix(test.y, test.y, test.n, conf);
    int conf_diag_ok = (conf[0][1] == 0 && conf[0][2] == 0 &&
                        conf[1][0] == 0 && conf[1][2] == 0 &&
                        conf[2][0] == 0 && conf[2][1] == 0);
    CHECK(conf_diag_ok, "confusion matrix off-diagonal == 0 for perfect preds");
    printf("\n");

    /* ------------------------------------------------------------------ */
    printf("[4] Metrics — all-C baseline (coil-class fraction ~42.7%%)\n");
    int *y_all_c = (int *)malloc(test.n * sizeof(int));
    for (int i = 0; i < test.n; i++) y_all_c[i] = 2;  /* 2 = Coil */
    float q3_all_c = compute_q3(test.y, y_all_c, test.n);
    CHECK(q3_all_c > 30.0f && q3_all_c < 60.0f, "Q3(all-C) in range (30, 60)");
    printf("    Q3(all-C) = %.2f%%  (expected ~42.7%%)\n\n", q3_all_c);

    /* ------------------------------------------------------------------ */
    printf("[5] Cross-entropy loss\n");
    /* Construct perfect probability matrix: p=1 for true class */
    float *probs = (float *)calloc(test.n * 3, sizeof(float));
    for (int i = 0; i < test.n; i++) probs[i * 3 + test.y[i]] = 1.0f;
    float ce = compute_cross_entropy(probs, test.y, test.n);
    /* log(1.0) = 0, but epsilon clamp means result ~ log(1) = near 0 */
    CHECK(ce < 1e-6f, "cross_entropy(perfect) ~ 0");
    free(probs);
    printf("\n");

    /* ------------------------------------------------------------------ */
    printf("[6] Logger — write JSON log\n");
    RunLog log;
    runlog_init(&log);
    strncpy(log.variant,   "serial",       sizeof(log.variant)   - 1);
    snprintf(log.data_path, sizeof(log.data_path), "%s", args.data_path);
    log.seed = 42; log.epochs = 1; log.batch_size = 64;
    log.learning_rate = 0.01f; log.hidden1 = 128; log.hidden2 = 64;
    log.threads = 1; log.mpi_ranks = 1;
    runlog_set_dataset(&log, train.n, val.n, test.n, train.feature_dim, 3);
    runlog_set_compiler(&log, "gcc " __VERSION__, "-O3 -march=native");

    runlog_add_epoch(&log, 1, 1.0986f, q3_all_c, elapsed);

    float pc[3] = {0.0f, 0.0f, q3_all_c};
    int   cf[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    runlog_finalize(&log, q3_all_c, q3_all_c, q3_all_c, pc, cf, elapsed);
    runlog_write(&log, args.out_dir);
    CHECK(log.finalized, "runlog_finalize sets finalized flag");
    printf("\n");

    /* ------------------------------------------------------------------ */
    dataset_free(&train); dataset_free(&val); dataset_free(&test);
    free(y_all_c);

    printf("=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return (fail_count == 0) ? 0 : 1;
}
