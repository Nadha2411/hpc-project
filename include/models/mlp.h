#ifndef MLP_H
#define MLP_H

/* Two-hidden-layer MLP for protein secondary structure prediction (Q3).
 *
 * Architecture:  Input(260) -> Hidden1(ReLU) -> Hidden2(ReLU) -> Output(3, Softmax)
 * Loss:          Cross-entropy (combined with softmax in backward pass)
 * Optimiser:     Mini-batch SGD
 *
 * All weight matrices are stored row-major:
 *   W1[hidden1][input_dim]   W2[hidden2][hidden1]   W3[output_dim][hidden2]
 *
 * Parallel variants (OpenMP, Pthreads, MPI) reuse this header unchanged.
 * Each thread/rank uses its own MLPGrad for gradient accumulation, then
 * reduces before calling mlp_sgd_update().
 */

/* -----------------------------------------------------------------------
 * Model weights
 * ----------------------------------------------------------------------- */
typedef struct {
    int input_dim;
    int hidden1;
    int hidden2;
    int output_dim;

    float *W1;   /* [hidden1  x input_dim]  */
    float *b1;   /* [hidden1]               */
    float *W2;   /* [hidden2  x hidden1]    */
    float *b2;   /* [hidden2]               */
    float *W3;   /* [output_dim x hidden2]  */
    float *b3;   /* [output_dim]            */
} MLP;

/* -----------------------------------------------------------------------
 * Gradient accumulators — one set per batch (zeroed before each batch).
 * In parallel variants each thread gets its own MLPGrad; gradients are
 * reduced before the SGD update.
 * ----------------------------------------------------------------------- */
typedef struct {
    float *dW1;  /* [hidden1  x input_dim]  */
    float *db1;  /* [hidden1]               */
    float *dW2;  /* [hidden2  x hidden1]    */
    float *db2;  /* [hidden2]               */
    float *dW3;  /* [output_dim x hidden2]  */
    float *db3;  /* [output_dim]            */
} MLPGrad;

/* -----------------------------------------------------------------------
 * MLP lifecycle
 * ----------------------------------------------------------------------- */

/* Allocate and He-initialise weights (W ~ Uniform(-sqrt(6/fan_in), +sqrt(6/fan_in))).
 * Biases set to 0. Uses a simple LCG seeded with 'seed'. */
void mlp_init(MLP *m, int input_dim, int hidden1, int hidden2,
              int output_dim, unsigned int seed);

/* Free all weight arrays. */
void mlp_free(MLP *m);

/* -----------------------------------------------------------------------
 * Gradient buffer lifecycle
 * ----------------------------------------------------------------------- */
void mlpgrad_alloc(MLPGrad *g, const MLP *m);
void mlpgrad_free(MLPGrad *g);

/* Zero all gradient accumulators (call before each batch). */
void mlpgrad_zero(MLPGrad *g, const MLP *m);

/* -----------------------------------------------------------------------
 * Forward pass (ONE sample)
 *
 * x       : [input_dim]   input features (read-only)
 * a1      : [hidden1]     post-ReLU activations, layer 1  (scratch, written)
 * a2      : [hidden2]     post-ReLU activations, layer 2  (scratch, written)
 * logits  : [output_dim]  raw pre-softmax scores          (scratch, written)
 * probs   : [output_dim]  softmax probabilities           (output, written)
 *
 * Caller must allocate a1, a2, logits, probs once and reuse across samples.
 * a1 and a2 must be preserved between forward() and backward() for the same sample.
 * ----------------------------------------------------------------------- */
void mlp_forward(const MLP *m,
                 const float *x,
                 float *a1, float *a2,
                 float *logits, float *probs);

/* -----------------------------------------------------------------------
 * Backward pass (ONE sample) — ADDS gradients into g.
 * Call mlpgrad_zero() before the first sample of each batch.
 *
 * x, a1, a2, probs : must be the values saved from the same forward() call.
 * y_true           : integer class label (0=H, 1=E, 2=C).
 *
 * Uses combined softmax + cross-entropy gradient:
 *   dL/dz3 = probs - one_hot(y_true)
 * ----------------------------------------------------------------------- */
void mlp_backward(const MLP *m, MLPGrad *g,
                  const float *x,
                  const float *a1, const float *a2,
                  const float *probs, int y_true);

/* -----------------------------------------------------------------------
 * SGD weight update (after one batch).
 * w -= (lr / batch_size) * dw   for all weight matrices and biases.
 * ----------------------------------------------------------------------- */
void mlp_sgd_update(MLP *m, const MLPGrad *g, float lr, int batch_size);

/* -----------------------------------------------------------------------
 * Batch inference (n samples).
 * X       : [n x feature_dim] row-major
 * y_pred  : [n]               filled with argmax class ids (0/1/2)
 * ----------------------------------------------------------------------- */
void mlp_predict(const MLP *m,
                 const float *X, int n, int feature_dim,
                 int *y_pred);

#endif /* MLP_H */
