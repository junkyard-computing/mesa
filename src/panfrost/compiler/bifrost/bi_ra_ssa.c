/*
 * Copyright 2023-2024 Alyssa Rosenzweig
 * Copyright 2023-2024 Valve Corporation
 * Copyright 2022 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */
#include "util/list.h"
#include "util/set.h"
#include "util/u_memory.h"
#include "bifrost_compile.h"
#include "bifrost_nir.h"
#include "compiler.h"

/*
 * RA treats the nesting counter, the divergent shuffle temporary, and the
 * spiller temporaries as alive throughout if used anywhere. This could be
 * optimized. Using a single power-of-two reserved region at the start ensures
 * these registers are never shuffled.
 */
static unsigned
reserved_size(bi_context *ctx)
{
   if (ctx->has_spill_pcopy_reserved)
      return 8;
   else
      return 0;
}

/*
 * Calculate register demand in registers, while gathering widths and
 * classes. Becuase we allocate in SSA, this calculation is exact in
 * linear-time. Depends on SSA liveness information.
 */
unsigned
bi_calc_register_demand(bi_context *ctx)
{
   /* Print detailed demand calculation, helpful to debug spilling */
   bool debug = false;

   if (debug) {
      bi_print_shader(ctx, stdout);
   }

   uint8_t *widths = calloc(ctx->ssa_alloc, sizeof(uint8_t));
   enum ra_class *classes = calloc(ctx->ssa_alloc, sizeof(enum ra_class));

   bi_foreach_instr_global(ctx, I) {
      bi_foreach_ssa_dest(I, d) {
         unsigned v = I->dest[d].value;
         assert(widths[v] == 0 && "broken SSA");
         /* Round up vectors for easier live range splitting */
         widths[v] = bi_count_write_registers(I, d);
         classes[v] = ra_class_for_index(I->dest[d]);
      }
   }
   /* now that we know the rest of the sizes, find the sizes for PHI nodes */
   bi_foreach_block(ctx, block) {
      bi_foreach_phi_in_block(block, I) {
         if (I->dest[0].type != BI_INDEX_NORMAL)
            continue;
         unsigned idx = I->dest[0].value;
         widths[idx] = 1;
         bi_foreach_ssa_src(I, s) {
            widths[idx] = MAX2(widths[idx], widths[I->src[s].value]);
         }
      }
   }

   /* Calculate demand at the start of each block based on live-in, then update
    * for each instruction processed. Calculate rolling maximum.
    */
   unsigned max_demand = 0;

   bi_foreach_block(ctx, block) {
      unsigned demand = reserved_size(ctx);

      /* Everything live-in */
      {
         int i;
         BITSET_FOREACH_SET(i, block->ssa_live_in, ctx->ssa_alloc) {
            if (classes[i] == RA_GPR)
               demand += widths[i];
         }
      }

      max_demand = MAX2(demand, max_demand);

      /* To handle non-power-of-two vectors, sometimes live range splitting
       * needs extra registers for 1 instruction. This counter tracks the number
       * of registers to be freed after 1 extra instruction.
       */
      unsigned late_kill_count = 0;

      if (debug) {
         printf("\n");
      }

      bi_foreach_instr_in_block(block, I) {
         /* Phis happen in parallel and are already accounted for in the live-in
          * set, just skip them so we don't double count.
          */
         if (I->op == BI_OPCODE_PHI)
            continue;

         if (debug) {
            printf("%u: ", demand);
            bi_print_instr(I, stdout);
         }

         /* Handle late-kill registers from last instruction */
         demand -= late_kill_count;
         late_kill_count = 0;

         /* Kill sources the first time we see them */
         bi_foreach_src(I, s) {
            if (!I->src[s].kill_ssa)
               continue;
            assert(I->src[s].type == BI_INDEX_NORMAL);
            if (ra_class_for_index(I->src[s]) != RA_GPR)
               continue;

            bool skip = false;

            for (unsigned backwards = 0; backwards < s; ++backwards) {
               if (bi_is_equiv(I->src[backwards], I->src[s])) {
                  skip = true;
                  break;
               }
            }

            if (!skip)
               demand -= widths[I->src[s].value];
         }

         /* Make destinations live */
         bi_foreach_ssa_dest(I, d) {
            if (ra_class_for_index(I->dest[d]) != RA_GPR)
               continue;

            /* Live range splits allocate at power-of-two granularity. Round up
             * destination sizes (temporarily) to powers-of-two.
             */
            unsigned real_width = widths[I->dest[d].value];
            unsigned pot_width = util_next_power_of_two(real_width);

            demand += pot_width;
            late_kill_count += (pot_width - real_width);
         }

         max_demand = MAX2(demand, max_demand);
      }

      demand -= late_kill_count;
   }

   free(widths);
   free(classes);
   return max_demand;
}

static bool
bi_is_fill(const bi_instr *I)
{
   return I->op == BI_OPCODE_MEMMOV && I->src[0].memory &&
          !I->dest[0].memory;
}

/*
 * The spiller inserts each fill immediately before its use, so the use waits
 * for the full TLS latency, and with every fill of a block allocated to the
 * same register there is nothing to overlap it with. Occupancy is at its
 * lowest precisely when we spill, so other warps can't hide it either.
 *
 * Hoist fills up to max_dist instructions earlier while the register demand
 * of every instruction the fill is moved across stays within k. A fill only
 * reads its spill slot, which is an SSA memory value, so the only other
 * barrier is the instruction defining that slot. Depends on SSA liveness
 * information, and keeps it valid.
 */
void
bi_hoist_fills(bi_context *ctx, unsigned k, unsigned max_dist)
{
   uint8_t *widths = calloc(ctx->ssa_alloc, sizeof(uint8_t));
   enum ra_class *classes = calloc(ctx->ssa_alloc, sizeof(enum ra_class));

   bi_foreach_instr_global(ctx, I) {
      bi_foreach_ssa_dest(I, d) {
         widths[I->dest[d].value] = bi_count_write_registers(I, d);
         classes[I->dest[d].value] = ra_class_for_index(I->dest[d]);
      }
   }

   bi_foreach_block(ctx, block) {
      bi_foreach_phi_in_block(block, I) {
         if (I->dest[0].type != BI_INDEX_NORMAL)
            continue;
         unsigned idx = I->dest[0].value;
         widths[idx] = 1;
         bi_foreach_ssa_src(I, s)
            widths[idx] = MAX2(widths[idx], widths[I->src[s].value]);
      }
   }

   bi_foreach_block(ctx, block) {
      unsigned n = 0;
      bi_foreach_instr_in_block(block, I)
         n++;

      bi_instr **ins = malloc(n * sizeof(*ins));
      unsigned *demand_at = malloc(n * sizeof(*demand_at));

      /* Same accounting as bi_calc_register_demand, recorded per instruction */
      unsigned demand = reserved_size(ctx);
      int i;
      BITSET_FOREACH_SET(i, block->ssa_live_in, ctx->ssa_alloc) {
         if (classes[i] == RA_GPR)
            demand += widths[i];
      }

      unsigned late_kill_count = 0;
      n = 0;
      bi_foreach_instr_in_block(block, I) {
         ins[n] = I;

         if (I->op == BI_OPCODE_PHI) {
            demand_at[n++] = demand;
            continue;
         }

         demand -= late_kill_count;
         late_kill_count = 0;

         bi_foreach_src(I, s) {
            if (!I->src[s].kill_ssa || ra_class_for_index(I->src[s]) != RA_GPR)
               continue;

            bool skip = false;
            for (unsigned b = 0; b < s; ++b)
               skip |= bi_is_equiv(I->src[b], I->src[s]);

            if (!skip)
               demand -= widths[I->src[s].value];
         }

         bi_foreach_ssa_dest(I, d) {
            if (ra_class_for_index(I->dest[d]) != RA_GPR)
               continue;

            unsigned real_width = widths[I->dest[d].value];
            unsigned pot_width = util_next_power_of_two(real_width);
            demand += pot_width;
            late_kill_count += pot_width - real_width;
         }

         demand_at[n++] = demand;
      }

      for (unsigned p = 0; p < n; ++p) {
         bi_instr *F = ins[p];
         if (!bi_is_fill(F))
            continue;

         unsigned w = util_next_power_of_two(widths[F->dest[0].value]);
         unsigned q = p;

         while (q > 0 && p - q < max_dist) {
            bi_instr *prev = ins[q - 1];

            if (prev->op == BI_OPCODE_PHI)
               break;

            bool defines_slot = false;
            bi_foreach_dest(prev, d)
               defines_slot |= bi_is_equiv(prev->dest[d], F->src[0]);

            if (defines_slot || demand_at[q - 1] + w > k)
               break;

            q--;
         }

         if (q == p)
            continue;

         list_del(&F->link);
         list_addtail(&F->link, &ins[q]->link);

         /* The fill's destination is now live across ins[q..p-1] */
         for (unsigned j = q; j < p; ++j)
            demand_at[j] += w;

         unsigned fill_demand = demand_at[p];
         memmove(&ins[q + 1], &ins[q], (p - q) * sizeof(*ins));
         memmove(&demand_at[q + 1], &demand_at[q], (p - q) * sizeof(*demand_at));
         ins[q] = F;
         demand_at[q] = (q > 0 ? demand_at[q - 1] : fill_demand) + w;
      }

      free(ins);
      free(demand_at);
   }

   free(widths);
   free(classes);
}
