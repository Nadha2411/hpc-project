#ifndef TIMER_H
#define TIMER_H

/* Wall-clock timer using CLOCK_MONOTONIC (Linux, MPI-safe).
 *
 * Usage:
 *   Timer t;
 *   timer_start(&t);
 *   // ... work ...
 *   double elapsed = timer_elapsed_s(&t);
 */

#include <time.h>

typedef struct {
    struct timespec start;
} Timer;

/* Start (or restart) the timer. */
void   timer_start(Timer *t);

/* Return seconds elapsed since timer_start(). */
double timer_elapsed_s(const Timer *t);

#endif /* TIMER_H */
