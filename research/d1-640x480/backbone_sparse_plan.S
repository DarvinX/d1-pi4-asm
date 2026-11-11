/* Select ranked image positions, and keep all text and causal tap support. */
#ifdef D1_SPARSE_VISUAL
.arch armv8-a
.include "src/common.inc"
.text
.balign 16
.global d1_sparse_plan
.global d1_sparse_plan_ranked
.global d1_sparse_plan_ranked12
.global d1_sparse_plan16
.type d1_sparse_plan,%function
.type d1_sparse_plan_ranked,%function
.type d1_sparse_plan_ranked12,%function
.type d1_sparse_plan16,%function
d1_sparse_plan: /* mass[300], first, positions, mask, active, support, counts */
    mov x7,xzr
d1_sparse_plan_ranked: /* x7 optionally receives the top-20 image mask */
    mov w8,#28
    mov w9,#48
    b .Lplan
d1_sparse_plan_ranked12: /* x7 receives the top-12 image mask */
    mov w8,#36
    mov w9,#48
    b .Lplan
.ifdef D1_BACKBONE_TRANSITION_FFN
.global d1_sparse_plan_budget8
.type d1_sparse_plan_budget8,%function
d1_sparse_plan_budget8: /* w9 active image budget, x7 receives the top eight */
    sub w8,w9,#8
    b .Lplan
.endif
.ifdef D1_BACKBONE_REDUCED_ROWS
.global d1_sparse_plan_budget_anchors
.type d1_sparse_plan_budget_anchors,%function
d1_sparse_plan_budget_anchors: /* w9 active image budget, w8 FFN anchor count */
    sub w8,w9,w8
    b .Lplan
.endif
d1_sparse_plan16: /* 16 active image positions, x7 receives the top eight */
    mov w8,#8
    mov w9,#16
.Lplan:
    ENTER
    fmov s2,w8
    mov x19,x0
    mov w20,w1
    mov w21,w2
    mov x22,x3
    mov x23,x4
    mov x24,x5
    mov x25,x6
    mov w27,w9
    mov x28,x7
    mov w8,#0
    add w9,w20,#300
    .ifdef D1_BACKBONE_DELAY_TEXT
    mov w14,w9
    ADDR x11,d1_sparse_text_live_tail
    ldr w11,[x11]
    cbz w11,.Lmask_init
    sub w14,w21,w11
    cmp w14,w9
    csel w14,w14,w9,hs
    .endif
.Lmask_init:
    cmp w8,w20
    cset w10,lo
    .ifdef D1_BACKBONE_DELAY_TEXT
    cmp w8,w14
    .else
    cmp w8,w9
    .endif
    cset w11,hs
    orr w10,w10,w11
    strb w10,[x22,x8]
    cbz x28,.Lmask_next
    cbnz w10,.Lmask_next
    .ifdef D1_BACKBONE_DELAY_TEXT
    cmp w8,w9
    b.hs .Lmask_next
    .endif
    sub w11,w8,w20
    strb wzr,[x28,x11]
.Lmask_next:
    add w8,w8,#1
    cmp w8,w21
    b.lo .Lmask_init
    add x26,x22,x20
.Lpick_row:
    mov w8,#0
    mov w9,#0
    mov w10,#0xff800000
    fmov s0,w10
.Lrank_scan:
    ldrb w10,[x26,x8]
    cbnz w10,.Lrank_next
    ldr s1,[x19,x8,lsl #2]
    fcmp s1,s0
    csel w9,w8,w9,gt
    fcsel s0,s1,s0,gt
.Lrank_next:
    add w8,w8,#1
    cmp w8,#300
    b.lo .Lrank_scan
    mov w10,#1
    strb w10,[x26,x9]
    cbz x28,.Lrank_done
    fmov w11,s2
    cmp w27,w11
    b.ls .Lrank_done
    strb w10,[x28,x9]
.Lrank_done:
    subs w27,w27,#1
    b.ne .Lpick_row

    mov w8,#0
    mov w9,#0 /* active count */
    mov w10,#0 /* support count */
.Llive_row:
    ldrb w11,[x22,x8]
    cbz w11,.Ltap_bounds
    str w8,[x23,x9,lsl #2]
    add w9,w9,#1
.Ltap_bounds:
    add w12,w8,#7
    cmp w12,w21
    csel w12,w12,w21,lo
    mov w13,w8
.Ltap_scan:
    ldrb w11,[x22,x13]
    cbnz w11,.Lsupport_row
    add w13,w13,#1
    cmp w13,w12
    b.lo .Ltap_scan
    b .Llive_next
.Lsupport_row:
    str w8,[x24,x10,lsl #2]
    add w10,w10,#1
.Llive_next:
    add w8,w8,#1
    cmp w8,w21
    b.lo .Llive_row
    stp w9,w10,[x25]
    LEAVE
.size d1_sparse_plan,.-d1_sparse_plan
.size d1_sparse_plan_ranked,.-d1_sparse_plan_ranked
.size d1_sparse_plan_ranked12,.-d1_sparse_plan_ranked12
.size d1_sparse_plan16,.-d1_sparse_plan16
.ifdef D1_BACKBONE_REDUCED_ROWS
.size d1_sparse_plan_budget_anchors,.-d1_sparse_plan_budget_anchors
.endif
.ifdef D1_BACKBONE_TRANSITION_FFN
.size d1_sparse_plan_budget8,.-d1_sparse_plan_budget8
.endif
#endif
.section .note.GNU-stack,"",%progbits
