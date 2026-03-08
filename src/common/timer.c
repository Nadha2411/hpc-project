#include "common/timer.h"

void timer_start(Timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

double timer_elapsed_s(const Timer *t) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec  - t->start.tv_sec)
         + (double)(now.tv_nsec - t->start.tv_nsec) * 1e-9;
}
