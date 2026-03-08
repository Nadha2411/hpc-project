#ifndef METRICS_H
#define METRICS_H

/* Evaluation metrics for 3-class protein secondary structure prediction.
 *
 * Class encoding (matches binary_meta.json):
 *   0 = H (Helix)
 *   1 = E (Sheet)
 *   2 = C (Coil)
 *
 * Paper reference: Zhong et al. (2007), Sect. 4
 *   Q3 = (correct_H + correct_E + correct_C) / total_residues * 100
 */

/* -----------------------------------------------------------------------
 * Q3 accuracy (paper-exact formula).
 * Returns value in [0, 100].
 * y_true, y_pred: arrays of length n with values in {0, 1, 2}.
 * ----------------------------------------------------------------------- */
float compute_q3(const int *y_true, const int *y_pred, int n);

/* -----------------------------------------------------------------------
 * Per-class accuracy.
 * out_acc[0] = H accuracy, out_acc[1] = E accuracy, out_acc[2] = C accuracy.
 * Each value in [0, 100].
 * ----------------------------------------------------------------------- */
void compute_per_class_accuracy(const int *y_true, const int *y_pred,
                                int n, float out_acc[3]);

/* -----------------------------------------------------------------------
 * Confusion matrix.
 * conf[true_class][pred_class] = count.
 * Rows = true label, Cols = predicted label. Order: H=0, E=1, C=2.
 * ----------------------------------------------------------------------- */
void compute_confusion_matrix(const int *y_true, const int *y_pred,
                               int n, int conf[3][3]);

/* -----------------------------------------------------------------------
 * Average cross-entropy loss over n samples.
 * probs: [n x 3] row-major softmax probabilities (each row sums to 1).
 * y_true: [n] ground-truth class ids.
 * A small epsilon (1e-9) is added to avoid log(0).
 * ----------------------------------------------------------------------- */
float compute_cross_entropy(const float *probs, const int *y_true, int n);

#endif /* METRICS_H */
