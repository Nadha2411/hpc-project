/* src/openmp/train.c — OpenMP parallel MLP training for protein secondary structure (Q3).
 *
 * Parallel strategy:
 *   - Each OpenMP thread owns a private MLPGrad buffer + scratch (a1, a2, logits, probs).
 *   - Mini-batch samples are divided across threads using a parallel for loop.
 *   - After each mini-batch, per-thread gradients are reduced (summed) into thread-0's
 *     buffer, then a single mlp_sgd_update() call updates the shared model weights.
 *   - The model weights (MLP *m) are READ-ONLY during the parallel region; only
 *     thread-local MLPGrad buffers are written, so no locking is needed.
 *
 * Environment hints for best performance:
 *   OMP_NUM_THREADS=N ./train_omp --threads N ...
 *   OMP_PROC_BIND=true OMP_PLACES=cores
 *
 * Architecture: Input(260) -> Hidden1(ReLU) -> Hidden2(ReLU) -> Output(3, Softmax)
 * Usage:        ./train_omp [--data PATH] [--epochs N] [--batch N]
 *                           [--lr F] [--seed N] [--threads N]
 *                           [--hidden1 N] [--hidden2 N]
 *                           [--out DIR] [--verbose 0|1]
 */

#include "common/cli.h"
#include "common/data_loader.h"
#include "common/logger.h"
#include "common/metrics.h"
#include "common/timer.h"
#include "models/mlp.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Fisher-Yates shuffle with an LCG (same generator as serial/train.c).
 * ----------------------------------------------------------------------- */
static unsigned int shuf_state;
static void shuf_seed(unsigned int s) { shuf_state = s; }
static unsigned int shuf_rand(void) {
    shuf_state = shuf_state * 1664525u + 1013904223u;
    return shuf_state;
}
static void shuffle(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(shuf_rand() % (unsigned int)(i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* -----------------------------------------------------------------------
 * Reduce: accumulate src gradient into dst (element-wise addition).
 * Called serially after the parallel region.
 * ----------------------------------------------------------------------- */
static void grad_reduce(MLPGrad *dst, const MLPGrad *src, const MLP *m) {
    int H1 = m->hidden1, H2 = m->hidden2, IN = m->input_dim, OUT = m->output_dim;
    for (int i = 0; i < H1 * IN;  i++) dst->dW1[i] += src->dW1[i];
    for (int i = 0; i < H1;       i++) dst->db1[i] += src->db1[i];
    for (int i = 0; i < H2 * H1;  i++) dst->dW2[i] += src->dW2[i];
    for (int i = 0; i < H2;       i++) dst->db2[i] += src->db2[i];
    for (int i = 0; i < OUT * H2; i++) dst->dW3[i] += src->dW3[i];
    for (int i = 0; i < OUT;      i++) dst->db3[i] += src->db3[i];
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */
int main(int argc, char **argv) {
    Args args;
    parse_args(argc, argv, &args, "openmp");

    /* Apply thread count from CLI */
    omp_set_num_threads(args.threads);

    if (args.verbose) {
        printf("=== train_omp ===\n");
        printf("  omp_threads : %d\n", args.threads);
        print_args(&args);
    }

    /* --- Load data --- */
    Dataset train, val, test;
    dataset_load(&train, args.data_path, "train");
    dataset_load(&val,   args.data_path, "val");
    dataset_load(&test,  args.data_path, "test");

    printf("Loaded: train=%d  val=%d  test=%d  feature_dim=%d\n",
           train.n, val.n, test.n, train.feature_dim);

    /* --- Run log --- */
    RunLog log;
    runlog_init(&log);
    strncpy(log.variant, "openmp", sizeof(log.variant) - 1);
    snprintf(log.data_path, sizeof(log.data_path), "%s", args.data_path);
    log.seed          = args.seed;
    log.epochs        = args.epochs;
    log.batch_size    = args.batch_size;
    log.learning_rate = args.lr;
    log.hidden1       = args.hidden1;
    log.hidden2       = args.hidden2;
    log.threads       = args.threads;
    log.mpi_ranks     = 1;
    runlog_set_dataset(&log, train.n, val.n, test.n, train.feature_dim, 3);
    runlog_set_compiler(&log, "gcc " __VERSION__, "-O3 -march=native -fopenmp");

    /* --- Initialise shared model --- */
    MLP m;
    mlp_init(&m, train.feature_dim, args.hidden1, args.hidden2, 3,
             (unsigned int)args.seed);

    /* --- Per-thread gradient buffers (allocated once) --- */
    int nthreads = args.threads;
    MLPGrad *tgrads = (MLPGrad *)malloc((size_t)nthreads * sizeof(MLPGrad));
    if (!tgrads) { fprintf(stderr, "[train_omp] OOM\n"); exit(1); }
    for (int t = 0; t < nthreads; t++)
        mlpgrad_alloc(&tgrads[t], &m);

    /* Per-thread scratch: a1, a2, logits, probs */
    int H1 = args.hidden1, H2 = args.hidden2;
    float **ta1     = (float **)malloc((size_t)nthreads * sizeof(float *));
    float **ta2     = (float **)malloc((size_t)nthreads * sizeof(float *));
    float **tlogits = (float **)malloc((size_t)nthreads * sizeof(float *));
    float **tprobs  = (float **)malloc((size_t)nthreads * sizeof(float *));
    if (!ta1 || !ta2 || !tlogits || !tprobs) {
        fprintf(stderr, "[train_omp] OOM scratch\n"); exit(1);
    }
    for (int t = 0; t < nthreads; t++) {
        ta1[t]     = (float *)malloc((size_t)H1 * sizeof(float));
        ta2[t]     = (float *)malloc((size_t)H2 * sizeof(float));
        tlogits[t] = (float *)malloc(3 * sizeof(float));
        tprobs[t]  = (float *)malloc(3 * sizeof(float));
        if (!ta1[t] || !ta2[t] || !tlogits[t] || !tprobs[t]) {
            fprintf(stderr, "[train_omp] OOM scratch[%d]\n", t); exit(1);
        }
    }

    /* Prediction + index buffers (serial use only) */
    int *indices      = (int *)malloc((size_t)train.n * sizeof(int));
    int *y_pred_val   = (int *)malloc((size_t)val.n   * sizeof(int));
    int *y_pred_train = (int *)malloc((size_t)train.n * sizeof(int));
    int *y_pred_test  = (int *)malloc((size_t)test.n  * sizeof(int));
    if (!indices || !y_pred_val || !y_pred_train || !y_pred_test) {
        fprintf(stderr, "[train_omp] OOM indices\n"); exit(1);
    }
    for (int i = 0; i < train.n; i++) indices[i] = i;

    /* --- Training loop --- */
    Timer total_timer;
    timer_start(&total_timer);

    float last_val_q3 = 0.0f;

    for (int epoch = 1; epoch <= args.epochs; epoch++) {
        /* Shuffle (serial — same as serial baseline for reproducibility) */
        shuf_seed((unsigned int)(args.seed + epoch));
        shuffle(indices, train.n);

        double epoch_loss = 0.0;
        Timer epoch_timer;
        timer_start(&epoch_timer);

        /* Mini-batch SGD */
        for (int b = 0; b < train.n; b += args.batch_size) {
            int actual = args.batch_size;
            if (b + actual > train.n) actual = train.n - b;

            /* Zero all per-thread gradient buffers */
            for (int t = 0; t < nthreads; t++)
                mlpgrad_zero(&tgrads[t], &m);

            double batch_loss = 0.0;

            /* Parallel: each thread processes a subset of the mini-batch */
            #pragma omp parallel reduction(+:batch_loss)
            {
                int tid = omp_get_thread_num();
                MLPGrad *g   = &tgrads[tid];
                float   *a1     = ta1[tid];
                float   *a2     = ta2[tid];
                float   *logits = tlogits[tid];
                float   *probs  = tprobs[tid];

                #pragma omp for schedule(static)
                for (int k = 0; k < actual; k++) {
                    int idx = indices[b + k];
                    const float *x = train.X + (size_t)idx * train.feature_dim;

                    mlp_forward(&m, x, a1, a2, logits, probs);

                    float p_true = probs[train.y[idx]];
                    if (p_true < 1e-9f) p_true = 1e-9f;
                    batch_loss -= (double)logf(p_true);

                    mlp_backward(&m, g, x, a1, a2, probs, train.y[idx]);
                }
            } /* end parallel */

            epoch_loss += batch_loss;

            /* Reduce thread gradients into tgrads[0] */
            for (int t = 1; t < nthreads; t++)
                grad_reduce(&tgrads[0], &tgrads[t], &m);

            /* Single SGD update using reduced gradient */
            mlp_sgd_update(&m, &tgrads[0], args.lr, actual);
        }

        double epoch_time = timer_elapsed_s(&epoch_timer);
        float avg_loss = (float)(epoch_loss / train.n);

        /* Validation Q3 */
        mlp_predict(&m, val.X, val.n, val.feature_dim, y_pred_val);
        float val_q3 = compute_q3(val.y, y_pred_val, val.n);
        last_val_q3 = val_q3;

        runlog_add_epoch(&log, epoch, avg_loss, val_q3, epoch_time);

        if (args.verbose) {
            printf("Epoch %3d  loss=%.4f  val_q3=%6.2f%%  lr=%.6f  %.2fs\n",
                   epoch, avg_loss, val_q3, args.lr, epoch_time);
        }

        /* LR step decay */
        if (args.lr_decay != 1.0f && epoch % args.lr_decay_every == 0)
            args.lr *= args.lr_decay;
    }

    /* --- Final evaluation --- */
    mlp_predict(&m, train.X, train.n, train.feature_dim, y_pred_train);
    mlp_predict(&m, test.X,  test.n,  test.feature_dim,  y_pred_test);

    float train_q3 = compute_q3(train.y, y_pred_train, train.n);
    float test_q3  = compute_q3(test.y,  y_pred_test,  test.n);
    float per_class[3];
    int   conf[3][3];
    compute_per_class_accuracy(test.y, y_pred_test, test.n, per_class);
    compute_confusion_matrix(test.y,  y_pred_test, test.n, conf);

    double total_s = timer_elapsed_s(&total_timer);
    runlog_finalize(&log, train_q3, last_val_q3, test_q3, per_class, conf, total_s);
    runlog_write(&log, args.out_dir);

    printf("\n=== Final Results ===\n");
    printf("threads   = %d\n", args.threads);
    printf("train Q3  = %.2f%%\n", train_q3);
    printf("val   Q3  = %.2f%%\n", last_val_q3);
    printf("test  Q3  = %.2f%%\n", test_q3);
    printf("per-class (H/E/C): %.2f%% / %.2f%% / %.2f%%\n",
           per_class[0], per_class[1], per_class[2]);
    printf("total time = %.1f s\n", total_s);

    /* --- Cleanup --- */
    for (int t = 0; t < nthreads; t++) {
        mlpgrad_free(&tgrads[t]);
        free(ta1[t]); free(ta2[t]); free(tlogits[t]); free(tprobs[t]);
    }
    free(tgrads); free(ta1); free(ta2); free(tlogits); free(tprobs);
    free(indices); free(y_pred_val); free(y_pred_train); free(y_pred_test);
    mlp_free(&m);
    dataset_free(&train);
    dataset_free(&val);
    dataset_free(&test);

    return 0;
}
