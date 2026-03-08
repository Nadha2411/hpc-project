/* src/serial/train.c — Stub entry point for serial training variant.
 *
 * Phase 3: This stub verifies that the Makefile, common headers, and
 * common libraries all compile and link correctly.
 *
 * Phase 4 will replace the body with the actual MLP training loop.
 */

#include "common/cli.h"
#include "common/data_loader.h"
#include "common/logger.h"
#include "common/metrics.h"
#include "common/timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    Args args;
    parse_args(argc, argv, &args, "serial");

    if (args.verbose) {
        printf("=== train_serial ===\n");
        print_args(&args);
    }

    /* --- Load data --- */
    Dataset train, val, test;
    dataset_load(&train, args.data_path, "train");
    dataset_load(&val,   args.data_path, "val");
    dataset_load(&test,  args.data_path, "test");

    printf("Loaded: train=%d  val=%d  test=%d  feature_dim=%d\n",
           train.n, val.n, test.n, train.feature_dim);

    /* --- Set up run log --- */
    RunLog log;
    runlog_init(&log);
    strncpy(log.variant,   "serial",        sizeof(log.variant)    - 1);
    snprintf(log.data_path, sizeof(log.data_path), "%s", args.data_path);
    log.seed          = args.seed;
    log.epochs        = args.epochs;
    log.batch_size    = args.batch_size;
    log.learning_rate = args.lr;
    log.hidden1       = args.hidden1;
    log.hidden2       = args.hidden2;
    log.threads       = 1;
    log.mpi_ranks     = 1;
    runlog_set_dataset(&log, train.n, val.n, test.n, train.feature_dim, 3);
    runlog_set_compiler(&log, "gcc " __VERSION__, "-O3 -march=native");

    /* --- Stub training (all-C baseline sanity check) --- */
    Timer total_timer;
    timer_start(&total_timer);

    /* Predict all-C (class 2) — known Q3 ≈ 42.7% (coil fraction) */
    int *y_pred_train = (int *)calloc(train.n, sizeof(int));
    int *y_pred_test  = (int *)calloc(test.n,  sizeof(int));
    for (int i = 0; i < train.n; i++) y_pred_train[i] = 2;
    for (int i = 0; i < test.n;  i++) y_pred_test[i]  = 2;

    float train_q3 = compute_q3(train.y, y_pred_train, train.n);
    float test_q3  = compute_q3(test.y,  y_pred_test,  test.n);
    float per_class[3]; int conf[3][3];
    compute_per_class_accuracy(test.y, y_pred_test, test.n, per_class);
    compute_confusion_matrix(test.y, y_pred_test, test.n, conf);

    /* Log one stub epoch */
    runlog_add_epoch(&log, 1, 1.0986f, train_q3, 0.001);

    double total_s = timer_elapsed_s(&total_timer);
    runlog_finalize(&log, train_q3, test_q3, test_q3, per_class, conf, total_s);
    runlog_write(&log, args.out_dir);

    printf("Stub Q3 (all-C baseline): train=%.2f%%  test=%.2f%%\n", train_q3, test_q3);
    printf("Expected ~42.7%% (coil fraction). Phase 4 will train real model.\n");

    free(y_pred_train);
    free(y_pred_test);
    dataset_free(&train);
    dataset_free(&val);
    dataset_free(&test);
    return 0;
}
