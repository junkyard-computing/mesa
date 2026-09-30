/*
 * Copyright 2026 Junkyard Computing
 * SPDX-License-Identifier: MIT
 */

#include "util/u_dynarray.h"
#include "bi_builder.h"
#include "compiler.h"

/*
 * Rematerialize cheap integer ALU chains near their uses.
 *
 * After full unrolling, CSE merges identical expressions across the whole
 * block. When such an expression is a short ALU chain over a few values that
 * stay live anyway, computing it once and keeping the result alive across a
 * long stretch of code trades a few ALU instructions for a register -- and
 * when many such results are live at once the spiller then pays for them with
 * a TLS spill and a fill per use.
 *
 * The typical case is dequantization in unrolled matrix multiplies: a handful
 * of packed words per row are unpacked into many operands that are each used
 * once per column of the other matrix. Keeping the packed words live and
 * unpacking next to each use needs far fewer registers than keeping every
 * unpacked operand live.
 *
 * So for a value produced by a cheap ALU chain whose uses span a long range,
 * clone the chain (down to its non-ALU leaves) in front of each later cluster
 * of uses. Runs after the last CSE and before scheduling and RA.
 */

#define REMAT_MAX_CHAIN  12
#define REMAT_MAX_LEAVES 4
#define REMAT_CLUSTER    24

static bool
bi_remat_cheap(const bi_instr *I)
{
   if (I->nr_dests != 1 || I->dest[0].type != BI_INDEX_NORMAL)
      return false;

   /* IMUL runs on the SFU at a quarter of the ALU rate on Valhall, so
    * recomputing one can cost more than the fill it replaces
    * (PAN_REMAT_NO_IMUL, measurement). */
   static int no_imul = -1;
   if (no_imul < 0)
      no_imul = getenv("PAN_REMAT_NO_IMUL") != NULL;
   if (no_imul && I->op == BI_OPCODE_IMUL_I32)
      return false;

   switch (I->op) {
   case BI_OPCODE_LSHIFT_AND_I32:
   case BI_OPCODE_LSHIFT_OR_I32:
   case BI_OPCODE_LSHIFT_XOR_I32:
   case BI_OPCODE_RSHIFT_AND_I32:
   case BI_OPCODE_RSHIFT_OR_I32:
   case BI_OPCODE_RSHIFT_XOR_I32:
   case BI_OPCODE_IMUL_I32:
   case BI_OPCODE_S16_TO_S32:
   case BI_OPCODE_U16_TO_U32:
   case BI_OPCODE_S8_TO_S32:
   case BI_OPCODE_U8_TO_U32:
   case BI_OPCODE_IADD_IMM_I32:
      break;
   default:
      return false;
   }

   bi_foreach_src(I, s) {
      enum bi_index_type t = I->src[s].type;
      if (t != BI_INDEX_NORMAL && t != BI_INDEX_CONSTANT &&
          t != BI_INDEX_FAU && t != BI_INDEX_NULL)
         return false;
      if (I->src[s].memory)
         return false;
   }

   return true;
}

struct remat_ctx {
   bi_context *shader;
   unsigned nr_values;   /* SSA values that existed before any cloning */
   bi_instr **def;       /* SSA value -> defining instruction */
   unsigned *pos;        /* SSA value -> position of its def in the block */
   bi_block **def_block; /* SSA value -> block of its def */
};

/* Collect the cheap ALU ancestors of I (including I) in the same block, in
 * definition order. Returns false if the chain is too large or has too many
 * leaves. */
static bool
bi_remat_collect(struct remat_ctx *rc, bi_block *block, bi_instr *I,
                 bi_instr **chain, unsigned *n, unsigned *leaves)
{
   for (unsigned i = 0; i < *n; ++i) {
      if (chain[i] == I)
         return true;
   }

   bi_foreach_ssa_src(I, s) {
      unsigned v = I->src[s].value;
      bi_instr *D = v < rc->nr_values ? rc->def[v] : NULL;

      if (D && rc->def_block[v] == block && bi_remat_cheap(D)) {
         if (!bi_remat_collect(rc, block, D, chain, n, leaves))
            return false;
      } else {
         if (++(*leaves) > REMAT_MAX_LEAVES)
            return false;
      }
   }

   if (*n >= REMAT_MAX_CHAIN)
      return false;

   chain[(*n)++] = I;
   return true;
}

static bi_instr *
bi_remat_clone(bi_builder *b, const bi_instr *I)
{
   size_t size =
      sizeof(bi_instr) + sizeof(bi_index) * (I->nr_dests + I->nr_srcs);
   bi_instr *J = rzalloc_size(b->shader, size);

   memcpy(J, I, size);
   J->dest = (bi_index *)(&J[1]);
   J->src = J->dest + J->nr_dests;
   J->dest[0] = bi_temp(b->shader);

   bi_foreach_src(J, s)
      J->src[s].kill_ssa = false;

   bi_builder_insert(&b->cursor, J);
   return J;
}

static bool
bi_remat_block(struct remat_ctx *rc, bi_block *block, unsigned min_span)
{
   bi_context *ctx = rc->shader;
   bool progress = false;

   /* Snapshot positions and uses before inserting anything. */
   struct util_dynarray instrs;
   util_dynarray_init(&instrs, NULL);

   unsigned n = 0;
   bi_foreach_instr_in_block(block, I) {
      util_dynarray_append(&instrs, I);
      bi_foreach_ssa_dest(I, d) {
         rc->pos[I->dest[d].value] = n;
      }
      n++;
   }

   if (n < 2 * min_span) {
      util_dynarray_fini(&instrs);
      return false;
   }

   bi_instr **ins = util_dynarray_begin(&instrs);

   /* Per value: positions of its uses in this block, and whether it has uses
    * elsewhere (then leave it alone). */
   const unsigned nr_values = rc->nr_values;
   struct util_dynarray *uses = calloc(nr_values, sizeof(*uses));
   bool *foreign = calloc(nr_values, sizeof(bool));

   bi_foreach_block(ctx, other) {
      bi_foreach_instr_in_block(other, I) {
         bi_foreach_ssa_src(I, s) {
            unsigned v = I->src[s].value;
            if (v >= nr_values || rc->def_block[v] != block)
               continue;
            if (other != block || I->op == BI_OPCODE_PHI)
               foreign[v] = true;
         }
      }
   }

   for (unsigned p = 0; p < n; ++p) {
      bi_foreach_ssa_src(ins[p], s) {
         unsigned v = ins[p]->src[s].value;
         if (v < nr_values && rc->def_block[v] == block) {
            unsigned use = p * 16 + s;
            util_dynarray_append(&uses[v], use);
         }
      }
   }

   /* TEMP tuning knob (not for upstream): PAN_REMAT_MAX=<n> only
    * rematerializes the n values with the longest use spans. */
   unsigned span_floor = min_span;
   const char *max_env = getenv("PAN_REMAT_MAX");
   if (max_env && atoi(max_env) > 0) {
      unsigned max_values = atoi(max_env);
      unsigned *spans = calloc(n, sizeof(unsigned));
      unsigned nspans = 0;

      for (unsigned p = 0; p < n; ++p) {
         if (!bi_remat_cheap(ins[p]))
            continue;
         unsigned v = ins[p]->dest[0].value;
         unsigned nu = v < nr_values
                          ? util_dynarray_num_elements(&uses[v], unsigned)
                          : 0;
         if (foreign[v] || nu < 2)
            continue;
         unsigned *u = util_dynarray_begin(&uses[v]);
         unsigned span = u[nu - 1] / 16 - u[0] / 16;
         if (span >= min_span)
            spans[nspans++] = span;
      }

      if (nspans > max_values) {
         /* Partial selection of the max_values-th largest span. */
         for (unsigned i = 0; i < max_values; ++i) {
            for (unsigned j = i + 1; j < nspans; ++j) {
               if (spans[j] > spans[i]) {
                  unsigned t = spans[i];
                  spans[i] = spans[j];
                  spans[j] = t;
               }
            }
         }
         span_floor = MAX2(min_span, spans[max_values - 1]);
      }

      free(spans);
   }

   for (unsigned p = 0; p < n; ++p) {
      bi_instr *V = ins[p];
      if (!bi_remat_cheap(V))
         continue;

      unsigned v = V->dest[0].value;
      if (v >= nr_values)
         continue;
      unsigned nu = util_dynarray_num_elements(&uses[v], unsigned);
      if (foreign[v] || nu < 2)
         continue;

      unsigned *u = util_dynarray_begin(&uses[v]);
      unsigned first = u[0] / 16, last = u[nu - 1] / 16;
      if (last - first < span_floor)
         continue;

      bi_instr *chain[REMAT_MAX_CHAIN];
      unsigned nchain = 0, leaves = 0;
      if (!bi_remat_collect(rc, block, V, chain, &nchain, &leaves))
         continue;

      /* The first cluster keeps the original. */
      unsigned cluster_start = first;
      bi_index cur = V->dest[0];

      for (unsigned k = 0; k < nu; ++k) {
         unsigned up = u[k] / 16, us = u[k] % 16;

         if (up - cluster_start >= REMAT_CLUSTER) {
            /* Clone the chain in front of this use. */
            bi_builder b = bi_init_builder(ctx, bi_before_instr(ins[up]));
            bi_index map_from[REMAT_MAX_CHAIN], map_to[REMAT_MAX_CHAIN];

            for (unsigned c = 0; c < nchain; ++c) {
               bi_instr *J = bi_remat_clone(&b, chain[c]);

               bi_foreach_ssa_src(J, s) {
                  for (unsigned m = 0; m < c; ++m) {
                     if (J->src[s].value == map_from[m].value)
                        J->src[s] = bi_replace_index(J->src[s], map_to[m]);
                  }
               }

               map_from[c] = chain[c]->dest[0];
               map_to[c] = J->dest[0];
            }

            cur = map_to[nchain - 1];
            cluster_start = up;
            progress = true;
         }

         if (cur.value != v)
            ins[up]->src[us] = bi_replace_index(ins[up]->src[us], cur);
      }
   }

   for (unsigned v = 0; v < nr_values; ++v)
      util_dynarray_fini(&uses[v]);
   free(uses);
   free(foreign);
   util_dynarray_fini(&instrs);
   return progress;
}

bool
bi_remat_alu_chains(bi_context *ctx, unsigned min_span)
{
   struct remat_ctx rc = {
      .shader = ctx,
      .nr_values = ctx->ssa_alloc,
      .def = calloc(ctx->ssa_alloc, sizeof(bi_instr *)),
      .pos = calloc(ctx->ssa_alloc, sizeof(unsigned)),
      .def_block = calloc(ctx->ssa_alloc, sizeof(bi_block *)),
   };
   bool progress = false;

   bi_foreach_block(ctx, block) {
      bi_foreach_instr_in_block(block, I) {
         bi_foreach_ssa_dest(I, d) {
            rc.def[I->dest[d].value] = I;
            rc.def_block[I->dest[d].value] = block;
         }
      }
   }

   /* Clones allocate new SSA values past the arrays above; they are never
    * looked up, since decisions only consider pre-existing instructions. */
   bi_foreach_block(ctx, block)
      progress |= bi_remat_block(&rc, block, min_span);

   free(rc.def);
   free(rc.pos);
   free(rc.def_block);
   return progress;
}
