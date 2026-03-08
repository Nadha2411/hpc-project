#include "common/metrics.h"

#include <math.h>
#include <string.h>

float compute_q3(const int *y_true, const int *y_pred, int n) {
    if (n <= 0) return 0.0f;
    int correct = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] == y_pred[i]) correct++;
    }
    return (float)correct / (float)n * 100.0f;
}

void compute_per_class_accuracy(const int *y_true, const int *y_pred,
                                int n, float out_acc[3]) {
    int total[3] = {0, 0, 0};
    int correct[3] = {0, 0, 0};

    for (int i = 0; i < n; i++) {
        int t = y_true[i];
        if (t >= 0 && t < 3) {
            total[t]++;
            if (y_pred[i] == t) correct[t]++;
        }
    }
    for (int c = 0; c < 3; c++) {
        out_acc[c] = (total[c] > 0)
                   ? (float)correct[c] / (float)total[c] * 100.0f
                   : 0.0f;
    }
}

void compute_confusion_matrix(const int *y_true, const int *y_pred,
                               int n, int conf[3][3]) {
    memset(conf, 0, 9 * sizeof(int));
    for (int i = 0; i < n; i++) {
        int t = y_true[i];
        int p = y_pred[i];
        if (t >= 0 && t < 3 && p >= 0 && p < 3) {
            conf[t][p]++;
        }
    }
}

float compute_cross_entropy(const float *probs, const int *y_true, int n) {
    if (n <= 0) return 0.0f;
    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        int c = y_true[i];
        /* probs row i starts at probs + i*3 */
        float p = probs[i * 3 + c];
        if (p < 1e-9f) p = 1e-9f;
        loss -= log((double)p);
    }
    return (float)(loss / (double)n);
}
