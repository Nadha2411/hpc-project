#include "common/cli.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog, const char *variant) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options (all variants share the same flags):\n"
        "  --data     <path>   Binary data directory  (default: data/processed/cb513/binary)\n"
        "  --epochs   <int>    Training epochs         (default: 50)\n"
        "  --batch    <int>    Batch size              (default: 64)\n"
        "  --lr       <float>  Learning rate           (default: 0.01)\n"
        "  --seed     <int>    RNG seed                (default: 42)\n"
        "  --threads  <int>    Threads (OMP/Pthreads)  (default: 1)\n"
        "  --hidden1  <int>    Hidden layer 1 size     (default: 128)\n"
        "  --hidden2  <int>    Hidden layer 2 size     (default: 64)\n"
        "  --out      <path>   Output directory        (default: results/%s)\n"
        "  --verbose  <0|1>    Print per-epoch info    (default: 0)\n"
        "  --help              Show this message\n",
        prog, variant);
}

void parse_args(int argc, char **argv, Args *args, const char *variant) {
    /* Defaults */
    strncpy(args->data_path, "data/processed/cb513/binary", sizeof(args->data_path) - 1);
    args->epochs    = 50;
    args->batch_size = 64;
    args->lr        = 0.01f;
    args->seed      = 42;
    args->threads   = 1;
    args->hidden1   = 128;
    args->hidden2   = 64;
    snprintf(args->out_dir, sizeof(args->out_dir), "results/%s", variant);
    args->verbose   = 0;

    static struct option long_opts[] = {
        {"data",    required_argument, 0, 'd'},
        {"epochs",  required_argument, 0, 'e'},
        {"batch",   required_argument, 0, 'b'},
        {"lr",      required_argument, 0, 'l'},
        {"seed",    required_argument, 0, 's'},
        {"threads", required_argument, 0, 't'},
        {"hidden1", required_argument, 0, '1'},
        {"hidden2", required_argument, 0, '2'},
        {"out",     required_argument, 0, 'o'},
        {"verbose", required_argument, 0, 'v'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt, idx = 0;
    while ((opt = getopt_long(argc, argv, "", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'd': strncpy(args->data_path, optarg, sizeof(args->data_path) - 1); break;
            case 'e': args->epochs     = atoi(optarg); break;
            case 'b': args->batch_size = atoi(optarg); break;
            case 'l': args->lr         = (float)atof(optarg); break;
            case 's': args->seed       = atoi(optarg); break;
            case 't': args->threads    = atoi(optarg); break;
            case '1': args->hidden1    = atoi(optarg); break;
            case '2': args->hidden2    = atoi(optarg); break;
            case 'o': strncpy(args->out_dir, optarg, sizeof(args->out_dir) - 1); break;
            case 'v': args->verbose    = atoi(optarg); break;
            case 'h': usage(argv[0], variant); exit(0);
            default:  usage(argv[0], variant); exit(1);
        }
    }
}

void print_args(const Args *args) {
    printf("  data_path : %s\n",  args->data_path);
    printf("  epochs    : %d\n",  args->epochs);
    printf("  batch     : %d\n",  args->batch_size);
    printf("  lr        : %.6f\n", args->lr);
    printf("  seed      : %d\n",  args->seed);
    printf("  threads   : %d\n",  args->threads);
    printf("  hidden1   : %d\n",  args->hidden1);
    printf("  hidden2   : %d\n",  args->hidden2);
    printf("  out_dir   : %s\n",  args->out_dir);
    printf("  verbose   : %d\n",  args->verbose);
}
