#include "models/mlp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =======================================================================
 * Internal: simple LCG random number generator (reproducible, no srand).
 * lcg_rand() returns a float in (-1, 1).
 * ======================================================================= */
static unsigned int lcg_state;

static void lcg_seed(unsigned int s) {
    lcg_state = s;
}

static float lcg_rand(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    /* Map upper 24 bits to [0, 1), then shift to (-1, 1) */
    return ((float)(lcg_state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}

/* =======================================================================
 * MLP lifecycle
 * ======================================================================= */

void mlp_init(MLP *m, int input_dim, int hidden1, int hidden2,
              int output_dim, unsigned int seed) {
    m->input_dim  = input_dim;
    m->hidden1    = hidden1;
    m->hidden2    = hidden2;
    m->output_dim = output_dim;

    m->W1 = (float *)malloc((size_t)hidden1    * (size_t)input_dim  * sizeof(float));
    m->b1 = (float *)calloc((size_t)hidden1,                          sizeof(float));
    m->W2 = (float *)malloc((size_t)hidden2    * (size_t)hidden1    * sizeof(float));
    m->b2 = (float *)calloc((size_t)hidden2,                          sizeof(float));
    m->W3 = (float *)malloc((size_t)output_dim * (size_t)hidden2    * sizeof(float));
    m->b3 = (float *)calloc((size_t)output_dim,                       sizeof(float));

    if (!m->W1 || !m->b1 || !m->W2 || !m->b2 || !m->W3 || !m->b3) {
        fprintf(stderr, "[mlp] OOM during mlp_init\n");
        exit(1);
    }

    lcg_seed(seed);

    /* He initialisation: W ~ Uniform(-sqrt(6/fan_in), +sqrt(6/fan_in)) */
    float scale1 = sqrtf(6.0f / (float)input_dim);
    for (int i = 0; i < hidden1 * input_dim; i++)
        m->W1[i] = lcg_rand() * scale1;

    float scale2 = sqrtf(6.0f / (float)hidden1);
    for (int i = 0; i < hidden2 * hidden1; i++)
        m->W2[i] = lcg_rand() * scale2;

    float scale3 = sqrtf(6.0f / (float)hidden2);
    for (int i = 0; i < output_dim * hidden2; i++)
        m->W3[i] = lcg_rand() * scale3;
    /* biases already zeroed by calloc */
}

void mlp_free(MLP *m) {
    free(m->W1); free(m->b1);
    free(m->W2); free(m->b2);
    free(m->W3); free(m->b3);
    m->W1 = m->b1 = m->W2 = m->b2 = m->W3 = m->b3 = NULL;
}

/* =======================================================================
 * Gradient buffer lifecycle
 * ======================================================================= */

void mlpgrad_alloc(MLPGrad *g, const MLP *m) {
    g->dW1 = (float *)malloc((size_t)m->hidden1    * (size_t)m->input_dim  * sizeof(float));
    g->db1 = (float *)malloc((size_t)m->hidden1                             * sizeof(float));
    g->dW2 = (float *)malloc((size_t)m->hidden2    * (size_t)m->hidden1    * sizeof(float));
    g->db2 = (float *)malloc((size_t)m->hidden2                             * sizeof(float));
    g->dW3 = (float *)malloc((size_t)m->output_dim * (size_t)m->hidden2    * sizeof(float));
    g->db3 = (float *)malloc((size_t)m->output_dim                          * sizeof(float));

    if (!g->dW1 || !g->db1 || !g->dW2 || !g->db2 || !g->dW3 || !g->db3) {
        fprintf(stderr, "[mlp] OOM during mlpgrad_alloc\n");
        exit(1);
    }
}

void mlpgrad_free(MLPGrad *g) {
    free(g->dW1); free(g->db1);
    free(g->dW2); free(g->db2);
    free(g->dW3); free(g->db3);
    g->dW1 = g->db1 = g->dW2 = g->db2 = g->dW3 = g->db3 = NULL;
}

void mlpgrad_zero(MLPGrad *g, const MLP *m) {
    memset(g->dW1, 0, (size_t)m->hidden1    * (size_t)m->input_dim  * sizeof(float));
    memset(g->db1, 0, (size_t)m->hidden1                             * sizeof(float));
    memset(g->dW2, 0, (size_t)m->hidden2    * (size_t)m->hidden1    * sizeof(float));
    memset(g->db2, 0, (size_t)m->hidden2                             * sizeof(float));
    memset(g->dW3, 0, (size_t)m->output_dim * (size_t)m->hidden2    * sizeof(float));
    memset(g->db3, 0, (size_t)m->output_dim                          * sizeof(float));
}

/* =======================================================================
 * Forward pass (one sample)
 * ======================================================================= */

void mlp_forward(const MLP *m,
                 const float *x,
                 float *a1, float *a2,
                 float *logits, float *probs) {
    int D  = m->input_dim;
    int H1 = m->hidden1;
    int H2 = m->hidden2;
    int K  = m->output_dim;

    /* Layer 1: z1 = W1 @ x + b1, a1 = ReLU(z1) */
    for (int i = 0; i < H1; i++) {
        float s = m->b1[i];
        const float *row = m->W1 + (size_t)i * D;
        for (int j = 0; j < D; j++) s += row[j] * x[j];
        a1[i] = (s > 0.0f) ? s : 0.0f;   /* ReLU */
    }

    /* Layer 2: z2 = W2 @ a1 + b2, a2 = ReLU(z2) */
    for (int i = 0; i < H2; i++) {
        float s = m->b2[i];
        const float *row = m->W2 + (size_t)i * H1;
        for (int j = 0; j < H1; j++) s += row[j] * a1[j];
        a2[i] = (s > 0.0f) ? s : 0.0f;   /* ReLU */
    }

    /* Output: logits = W3 @ a2 + b3 */
    for (int i = 0; i < K; i++) {
        float s = m->b3[i];
        const float *row = m->W3 + (size_t)i * H2;
        for (int j = 0; j < H2; j++) s += row[j] * a2[j];
        logits[i] = s;
    }

    /* Softmax (subtract max for numerical stability) */
    float max_l = logits[0];
    for (int i = 1; i < K; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < K; i++) { probs[i] = expf(logits[i] - max_l); sum += probs[i]; }
    for (int i = 0; i < K; i++) probs[i] /= sum;
}

/* =======================================================================
 * Backward pass (one sample) — ADDS to g
 * ======================================================================= */

void mlp_backward(const MLP *m, MLPGrad *g,
                  const float *x,
                  const float *a1, const float *a2,
                  const float *probs, int y_true) {
    int D  = m->input_dim;
    int H1 = m->hidden1;
    int H2 = m->hidden2;
    int K  = m->output_dim;

    /* --- dL/dz3 = probs - one_hot(y_true)  (CE + softmax combined) --- */
    float dz3[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < K; i++) dz3[i] = probs[i];
    dz3[y_true] -= 1.0f;

    /* dW3 += dz3 ⊗ a2,   db3 += dz3 */
    for (int i = 0; i < K; i++) {
        g->db3[i] += dz3[i];
        float *row = g->dW3 + (size_t)i * H2;
        for (int j = 0; j < H2; j++) row[j] += dz3[i] * a2[j];
    }

    /* da2 = W3^T @ dz3 */
    float *da2 = (float *)malloc((size_t)H2 * sizeof(float));
    float *dz2 = (float *)malloc((size_t)H2 * sizeof(float));
    if (!da2 || !dz2) { fprintf(stderr, "[mlp_backward] OOM\n"); exit(1); }
    for (int j = 0; j < H2; j++) {
        float s = 0.0f;
        for (int i = 0; i < K; i++) s += m->W3[(size_t)i * H2 + j] * dz3[i];
        da2[j] = s;
    }

    /* dz2 = da2 * ReLU'(a2)   (a2 > 0 was ReLU applied) */
    for (int j = 0; j < H2; j++) dz2[j] = (a2[j] > 0.0f) ? da2[j] : 0.0f;

    /* dW2 += dz2 ⊗ a1,   db2 += dz2 */
    for (int i = 0; i < H2; i++) {
        g->db2[i] += dz2[i];
        float *row = g->dW2 + (size_t)i * H1;
        for (int j = 0; j < H1; j++) row[j] += dz2[i] * a1[j];
    }

    /* da1 = W2^T @ dz2 */
    float *da1 = (float *)malloc((size_t)H1 * sizeof(float));
    if (!da1) { fprintf(stderr, "[mlp_backward] OOM\n"); exit(1); }
    for (int j = 0; j < H1; j++) {
        float s = 0.0f;
        for (int i = 0; i < H2; i++) s += m->W2[(size_t)i * H1 + j] * dz2[i];
        da1[j] = s;
    }

    /* dz1 = da1 * ReLU'(a1) */
    for (int j = 0; j < H1; j++) {
        float dz1j = (a1[j] > 0.0f) ? da1[j] : 0.0f;
        g->db1[j] += dz1j;
        float *row = g->dW1 + (size_t)j * D;
        for (int k = 0; k < D; k++) row[k] += dz1j * x[k];
    }

    free(da2); free(dz2);
    free(da1);
}

/* =======================================================================
 * SGD update: w -= (lr / batch_size) * dw
 * ======================================================================= */

void mlp_sgd_update(MLP *m, const MLPGrad *g, float lr, int batch_size) {
    float scale = lr / (float)batch_size;
    int H1 = m->hidden1, H2 = m->hidden2, D = m->input_dim, K = m->output_dim;

    for (int i = 0; i < H1 * D;  i++) m->W1[i] -= scale * g->dW1[i];
    for (int i = 0; i < H1;      i++) m->b1[i] -= scale * g->db1[i];
    for (int i = 0; i < H2 * H1; i++) m->W2[i] -= scale * g->dW2[i];
    for (int i = 0; i < H2;      i++) m->b2[i] -= scale * g->db2[i];
    for (int i = 0; i < K  * H2; i++) m->W3[i] -= scale * g->dW3[i];
    for (int i = 0; i < K;       i++) m->b3[i] -= scale * g->db3[i];
}

/* =======================================================================
 * Batch inference
 * ======================================================================= */

void mlp_predict(const MLP *m, const float *X, int n, int feature_dim,
                 int *y_pred) {
    /* Allocate scratch buffers once */
    float *a1     = (float *)malloc((size_t)m->hidden1    * sizeof(float));
    float *a2     = (float *)malloc((size_t)m->hidden2    * sizeof(float));
    float *logits = (float *)malloc((size_t)m->output_dim * sizeof(float));
    float *probs  = (float *)malloc((size_t)m->output_dim * sizeof(float));
    if (!a1 || !a2 || !logits || !probs) {
        fprintf(stderr, "[mlp_predict] OOM\n"); exit(1);
    }

    for (int i = 0; i < n; i++) {
        mlp_forward(m, X + (size_t)i * feature_dim, a1, a2, logits, probs);
        /* argmax */
        int best = 0;
        for (int c = 1; c < m->output_dim; c++)
            if (probs[c] > probs[best]) best = c;
        y_pred[i] = best;
    }

    free(a1); free(a2); free(logits); free(probs);
}
