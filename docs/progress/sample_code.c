#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AMINO_ACIDS 20
#define WINDOW_SIZE 13
#define HALF_WINDOW 6
#define FEATURE_DIM (WINDOW_SIZE * AMINO_ACIDS)

/*
    Protein Secondary Structure Prediction - Sample C Code

    This is a simple project-related demo program.
    It does NOT train the full neural network, but it shows the important
    preprocessing idea used in your project:

    1. Take a protein sequence
    2. Build a sliding window around each residue
    3. One-hot encode each amino acid
    4. Concatenate into a 260-dimensional feature vector
    5. Make a simple placeholder prediction (demo only)

    Amino acid order used for one-hot encoding:
    A C D E F G H I K L M N P Q R S T V W Y
*/

const char AA_ORDER[AMINO_ACIDS + 1] = "ACDEFGHIKLMNPQRSTVWY";

/* Convert amino acid letter to index in AA_ORDER */
int aa_to_index(char aa) {
    for (int i = 0; i < AMINO_ACIDS; i++) {
        if (AA_ORDER[i] == aa) {
            return i;
        }
    }
    return -1; /* unknown / padding */
}

/* Print one-hot vector for a single amino acid */
void print_one_hot(char aa) {
    int idx = aa_to_index(aa);

    printf("One-hot for '%c': [", aa);
    for (int i = 0; i < AMINO_ACIDS; i++) {
        if (idx == i) {
            printf("1");
        } else {
            printf("0");
        }

        if (i != AMINO_ACIDS - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

/* Create one-hot vector for a single amino acid */
void encode_amino_acid(char aa, float out[AMINO_ACIDS]) {
    for (int i = 0; i < AMINO_ACIDS; i++) {
        out[i] = 0.0f;
    }

    int idx = aa_to_index(aa);
    if (idx >= 0) {
        out[idx] = 1.0f;
    }
    /* else keep all zeros for padding/unknown */
}

/* Build sliding window centered at position pos */
void build_window(const char *sequence, int seq_len, int pos, char window[WINDOW_SIZE + 1]) {
    int start = pos - HALF_WINDOW;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        int seq_index = start + i;
        if (seq_index < 0 || seq_index >= seq_len) {
            window[i] = 'X'; /* padding */
        } else {
            window[i] = sequence[seq_index];
        }
    }

    window[WINDOW_SIZE] = '\0';
}

/* Encode the whole 13-residue window into 260 floats */
void encode_window(const char window[WINDOW_SIZE + 1], float features[FEATURE_DIM]) {
    for (int i = 0; i < WINDOW_SIZE; i++) {
        float temp[AMINO_ACIDS];
        encode_amino_acid(window[i], temp);

        for (int j = 0; j < AMINO_ACIDS; j++) {
            features[i * AMINO_ACIDS + j] = temp[j];
        }
    }
}

/* Print the window nicely */
void print_window(const char window[WINDOW_SIZE + 1], int center_pos) {
    printf("Window centered at residue position %d:\n", center_pos + 1);
    printf("[ ");
    for (int i = 0; i < WINDOW_SIZE; i++) {
        printf("%c ", window[i]);
    }
    printf("]\n");

    printf("  ");
    for (int i = 0; i < WINDOW_SIZE; i++) {
        if (i == HALF_WINDOW) {
            printf("^ ");
        } else {
            printf("  ");
        }
    }
    printf("\n");
}

/* Print only first n values of feature vector */
void print_feature_preview(const float features[FEATURE_DIM], int n) {
    printf("Feature vector preview (first %d of %d values):\n", n, FEATURE_DIM);
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%.0f", features[i]);
        if (i != n - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

/*
    Simple demo scoring function
    This is NOT a real MLP.
    It just uses crude counts to produce a mock prediction:
    - Helix-like amino acids: A, L, M, Q, E, K, H
    - Strand-like amino acids: V, I, Y, F, W, T
    - Others -> coil tendency
*/
char simple_demo_predict(const char window[WINDOW_SIZE + 1]) {
    int helix_score = 0;
    int strand_score = 0;
    int coil_score = 0;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        char aa = window[i];

        if (aa == 'X') {
            continue;
        }

        switch (aa) {
            case 'A':
            case 'L':
            case 'M':
            case 'Q':
            case 'E':
            case 'K':
            case 'H':
                helix_score++;
                break;

            case 'V':
            case 'I':
            case 'Y':
            case 'F':
            case 'W':
            case 'T':
                strand_score++;
                break;

            default:
                coil_score++;
                break;
        }
    }

    if (helix_score >= strand_score && helix_score >= coil_score) {
        return 'H';
    } else if (strand_score >= helix_score && strand_score >= coil_score) {
        return 'E';
    } else {
        return 'C';
    }
}

/* Predict structure for every residue in the sequence */
void predict_sequence(const char *sequence, char *predicted_structure) {
    int seq_len = (int)strlen(sequence);

    for (int pos = 0; pos < seq_len; pos++) {
        char window[WINDOW_SIZE + 1];
        build_window(sequence, seq_len, pos, window);
        predicted_structure[pos] = simple_demo_predict(window);
    }

    predicted_structure[seq_len] = '\0';
}

/* Print sequence with positions */
void print_sequence_with_positions(const char *label, const char *sequence) {
    int len = (int)strlen(sequence);

    printf("%s\n", label);
    printf("Pos : ");
    for (int i = 0; i < len; i++) {
        printf("%2d ", i + 1);
    }
    printf("\n");

    printf("Data: ");
    for (int i = 0; i < len; i++) {
        printf(" %c ", sequence[i]);
    }
    printf("\n");
}

/* Show one residue processing example */
void demo_single_residue_processing(const char *sequence, int pos) {
    int seq_len = (int)strlen(sequence);
    if (pos < 0 || pos >= seq_len) {
        printf("Invalid residue position.\n");
        return;
    }

    char window[WINDOW_SIZE + 1];
    float features[FEATURE_DIM];

    build_window(sequence, seq_len, pos, window);
    encode_window(window, features);

    printf("\n=== Single Residue Demo ===\n");
    printf("Target residue position: %d\n", pos + 1);
    printf("Target amino acid      : %c\n", sequence[pos]);

    print_window(window, pos);
    print_feature_preview(features, 40);

    char pred = simple_demo_predict(window);
    printf("Demo predicted secondary structure for residue %c at position %d = %c\n",
           sequence[pos], pos + 1, pred);
}

/* Main program */
int main(void) {
    const char *protein_sequence = "MKTAYIAKQRQGMPE";
    int seq_len = (int)strlen(protein_sequence);

    char *predicted_structure = (char *)malloc((seq_len + 1) * sizeof(char));
    if (predicted_structure == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    printf("===============================================\n");
    printf(" Protein Secondary Structure Prediction Demo\n");
    printf("===============================================\n\n");

    printf("Amino acid alphabet used for one-hot encoding:\n");
    printf("%s\n\n", AA_ORDER);

    print_one_hot('A');
    print_one_hot('G');
    print_one_hot('Y');

    print_sequence_with_positions("\nInput Protein Sequence:", protein_sequence);

    /* Show preprocessing example for one residue */
    demo_single_residue_processing(protein_sequence, 6); /* position 7 in human counting */

    /* Predict whole sequence */
    predict_sequence(protein_sequence, predicted_structure);

    printf("\n=== Whole Sequence Prediction ===\n");
    printf("Sequence : %s\n", protein_sequence);
    printf("Predicted: %s\n", predicted_structure);

    printf("\nLegend:\n");
    printf("H = Alpha Helix\n");
    printf("E = Beta Strand\n");
    printf("C = Coil\n");

    printf("\nNote:\n");
    printf("This is a preprocessing + demo prediction program only.\n");
    printf("It is related to your project, but it is not the full MLP trainer.\n");
    printf("Your real project uses 260-dim window features, neural network layers,\n");
    printf("forward pass, backpropagation, SGD, and parallel OpenMP training.\n");

    free(predicted_structure);
    return 0;
}