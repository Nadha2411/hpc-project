#ifndef CLI_H
#define CLI_H

/* Unified CLI argument contract for all training variants.
 *
 * All 5 binaries (serial, openmp, pthreads, mpi, hybrid) accept the same
 * flags. Flags irrelevant to a variant are silently ignored.
 *
 * Usage:
 *   Args args;
 *   parse_args(argc, argv, &args, "serial");
 *   print_args(&args);   // optional verbose dump
 */

typedef struct {
    char  data_path[256]; /* --data     default: data/processed/cb513/binary */
    int   epochs;         /* --epochs   default: 50  */
    int   batch_size;     /* --batch    default: 64  */
    float lr;             /* --lr       default: 0.01 */
    int   seed;           /* --seed     default: 42  */
    int   threads;        /* --threads  default: 1   (OpenMP/Pthreads only) */
    int   hidden1;        /* --hidden1  default: 128 */
    int   hidden2;        /* --hidden2  default: 64  */
    char  out_dir[256];   /* --out           default: results/<variant> */
    int   verbose;        /* --verbose       default: 0   */
    float lr_decay;       /* --lr-decay      default: 1.0 (no decay)  */
    int   lr_decay_every; /* --lr-decay-every default: 20 epochs      */
} Args;

/* Parse argc/argv into args using the given variant name for defaults.
 * Prints usage and exits on unknown flag or --help. */
void parse_args(int argc, char **argv, Args *args, const char *variant);

/* Print all argument values to stdout (useful with --verbose). */
void print_args(const Args *args);

#endif /* CLI_H */
