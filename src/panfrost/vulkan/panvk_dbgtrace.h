/*
 * LOCAL DEBUG (not for upstream): PANVK_TRACE_POOL=1 logs command-buffer
 * submits/resets and memory-pool BO recycling with CLOCK_MONOTONIC timestamps
 * (the clock dmesg uses), to find memory recycled while GPU work that uses it
 * may still be running.
 */
#ifndef PANVK_DBGTRACE_H
#define PANVK_DBGTRACE_H

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline bool
panvk_dbgtrace_on(void)
{
   static int on = -1;
   if (on < 0)
      on = getenv("PANVK_TRACE_POOL") != NULL;
   return on;
}

static inline void __attribute__((format(printf, 1, 2)))
panvk_dbgtrace(const char *fmt, ...)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   char buf[512];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);
   fprintf(stderr, "POOLTRACE %ld.%06ld %s\n", (long)ts.tv_sec,
           ts.tv_nsec / 1000, buf);
}

#endif
