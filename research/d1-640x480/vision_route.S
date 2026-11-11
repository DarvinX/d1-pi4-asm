/* Four spatial queries share a fixed refinement budget over 300 key quads. */
.arch armv8-a
.include "src/common.inc"

.equ ROUTE_GROUPS,300
.equ ROUTE_COARSE,0
.equ ROUTE_PRIORITY,4800
.equ ROUTE_MASK,6000
.equ ROUTE_TEMP,6304
.equ ROUTE_SCORES,6336
.equ ROUTE_SCORE_STRIDE,2688 /* 672 floats, at most 660 live */
.equ ROUTE_VALUE_PTRS,17088
.equ ROUTE_STACK_PAGES,6

.text
.balign 32
.global d1_route_bins
.type d1_route_bins,%function
/* x0: 300 finite scores; x1: 300 mask bytes; w2: budget in [1, 300]. */
d1_route_bins:
    sub sp,sp,#32
    ldr q0,[x0]
    mov v1.16b,v0.16b
    mov x3,x0
    mov w4,#ROUTE_GROUPS/4
.Lroute_range:
    ldr q2,[x3],#16
    fmin v0.4s,v0.4s,v2.4s
    fmax v1.4s,v1.4s,v2.4s
    subs w4,w4,#1
    b.ne .Lroute_range
    fminv s0,v0.4s
    fmaxv s1,v1.4s
    fsub s1,s1,s0
    fmov s2,#15.0
    fcmp s1,#0.0
    b.eq .Lroute_equal
    fdiv s1,s2,s1
    b .Lroute_range_ready
.Lroute_equal:
    fmov s1,wzr
.Lroute_range_ready:
    stp xzr,xzr,[sp]
    stp xzr,xzr,[sp,#16]
    mov w3,#0
    mov w7,#15
.Lroute_bucket:
    ldr s2,[x0,x3,lsl #2]
    fsub s2,s2,s0
    fmul s2,s2,s1
    fcvtzu w4,s2
    cmp w4,w7
    csel w4,w4,w7,ls
    strb w4,[x1,x3]
    ldrh w5,[sp,w4,uxtw #1]
    add w5,w5,#1
    strh w5,[sp,w4,uxtw #1]
    add w3,w3,#1
    cmp w3,#ROUTE_GROUPS
    b.lo .Lroute_bucket
    mov w4,w7
.Lroute_cutoff:
    ldrh w5,[sp,w4,uxtw #1]
    cmp w2,w5
    b.ls .Lroute_cutoff_ready
    sub w2,w2,w5
    sub w4,w4,#1
    b .Lroute_cutoff
.Lroute_cutoff_ready:
    mov w3,#0
.Lroute_mask:
    ldrb w5,[x1,x3]
    cmp w5,w4
    b.eq .Lroute_tie
    cset w6,hi
    b .Lroute_write_mask
.Lroute_tie:
    cmp w2,#0
    cset w6,ne
    sub w2,w2,w6
.Lroute_write_mask:
    strb w6,[x1,x3]
    add w3,w3,#1
    cmp w3,#ROUTE_GROUPS
    b.lo .Lroute_mask
    add sp,sp,#32
    ret
.size d1_route_bins,.-d1_route_bins

.macro ROUTE_VALUE_CHUNK a,b,c,d,lane
    fmla v\a\().4s,v0.4s,v\lane\().s[0]
    fmla v\b\().4s,v1.4s,v\lane\().s[0]
    fmla v\c\().4s,v2.4s,v\lane\().s[0]
    fmla v\d\().4s,v3.4s,v\lane\().s[0]
.endm

.type d1_route_values4,%function
/* x0 probabilities with fixed stride, x1 direct V row pointers, x2 out,
 * w3 mixed key count. Four contiguous 64-float output rows. */
d1_route_values4:
    mov x4,#0
.Lroute_value_channels:
    .irp acc,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    movi v\acc\().4s,#0
    .endr
    mov x8,x0
    add x9,x8,#ROUTE_SCORE_STRIDE
    add x10,x9,#ROUTE_SCORE_STRIDE
    add x11,x10,#ROUTE_SCORE_STRIDE
    mov x12,x1
    mov w13,w3
.Lroute_value_rows:
    ldr x5,[x12],#8
    add x5,x5,x4
    ld1 {v0.4s-v3.4s},[x5]
    ldr s4,[x8],#4
    ldr s5,[x9],#4
    ldr s6,[x10],#4
    ldr s7,[x11],#4
    ROUTE_VALUE_CHUNK 16,17,18,19,4
    ROUTE_VALUE_CHUNK 20,21,22,23,5
    ROUTE_VALUE_CHUNK 24,25,26,27,6
    ROUTE_VALUE_CHUNK 28,29,30,31,7
    subs w13,w13,#1
    b.ne .Lroute_value_rows
    add x5,x2,x4
    st1 {v16.4s-v19.4s},[x5]
    add x5,x5,#256
    st1 {v20.4s-v23.4s},[x5]
    add x5,x5,#256
    st1 {v24.4s-v27.4s},[x5]
    add x5,x5,#256
    st1 {v28.4s-v31.4s},[x5]
    add x4,x4,#64
    cmp x4,#256
    b.lo .Lroute_value_channels
    ret
.size d1_route_values4,.-d1_route_values4

.macro ROUTE_RECORD
    str x1,[x23,x25,lsl #3]
    fmax v10.4s,v10.4s,v0.4s
    add x0,x24,x25,lsl #2
    st1 {v0.s}[0],[x0]
    add x0,x0,#ROUTE_SCORE_STRIDE
    st1 {v0.s}[1],[x0]
    add x0,x0,#ROUTE_SCORE_STRIDE
    st1 {v0.s}[2],[x0]
    add x0,x0,#ROUTE_SCORE_STRIDE
    st1 {v0.s}[3],[x0]
    add x25,x25,#1
.endm

.balign 64
.global d1_route_attention4
.type d1_route_attention4,%function
/* struct: Q, quad-major K/V, mean K/V, radius, output, uint32 budget.
 * Q and output are [4][64]. Original rows are [300][4][64]. */
d1_route_attention4:
    ENTER
    sub sp,sp,#ROUTE_STACK_PAGES,lsl #12
    mov x19,x0
    ldr x20,[x19]
    ldr x21,[x19,#24]
    add x22,sp,#ROUTE_COARSE
    add x23,sp,#4096
    add x23,x23,#ROUTE_PRIORITY-4096
    ldr x24,[x19,#40]
    mov w25,#0
    fmov v9.4s,#0.125
    add x26,sp,#4096
    add x26,x26,#ROUTE_TEMP-4096
.Lroute_norm_row:
    add x0,x20,x25,lsl #8
    movi v0.4s,#0
    mov w1,#16
.Lroute_norm_vector:
    ldr q1,[x0],#16
    fmla v0.4s,v1.4s,v1.4s
    subs w1,w1,#1
    b.ne .Lroute_norm_vector
    faddp v0.4s,v0.4s,v0.4s
    faddp v0.4s,v0.4s,v0.4s
    fsqrt s0,s0
    fmul s0,s0,s9
    str s0,[x26,x25,lsl #2]
    add w25,w25,#1
    cmp w25,#4
    b.lo .Lroute_norm_row
    ldr q8,[x26]
    mov w25,#ROUTE_GROUPS
.Lroute_coarse_group:
    mov x0,x21
    mov x1,x20
    mov x2,#256
    mov x3,x22
    bl d1_dot4_f32_native64
    ldr q0,[x22]
    fmul v0.4s,v0.4s,v9.4s
    str q0,[x22],#16
    ld1r {v1.4s},[x24],#4
    fmla v0.4s,v8.4s,v1.4s
    fmaxv s0,v0.4s
    str s0,[x23],#4
    add x21,x21,#256
    subs w25,w25,#1
    b.ne .Lroute_coarse_group
    add x0,sp,#4096
    add x0,x0,#ROUTE_PRIORITY-4096
    add x26,sp,#4096
    add x26,x26,#ROUTE_MASK-4096
    mov x1,x26
    ldr w2,[x19,#56]
    bl d1_route_bins
    mov w21,#0
    add x23,sp,#4,lsl #12
    add x23,x23,#ROUTE_VALUE_PTRS-16384
    add x24,sp,#4096
    add x24,x24,#ROUTE_SCORES-4096
    mov x25,#0
    add x27,sp,#ROUTE_COARSE
    mov w0,#0xff800000
    dup v10.4s,w0
    ldr s0,.Lroute_log_four
    dup v8.4s,v0.s[0]
    add x28,sp,#4096
    add x28,x28,#ROUTE_TEMP-4096
.Lroute_mixed_group:
    ldrb w0,[x26,x21]
    cbnz w0,.Lroute_refine
    ldr q0,[x27,x21,lsl #4]
    fadd v0.4s,v0.4s,v8.4s
    ldr x1,[x19,#32]
    add x1,x1,x21,lsl #8
    ROUTE_RECORD
    b .Lroute_next_group
.Lroute_refine:
    mov w22,#0
.Lroute_fine_row:
    ldr x0,[x19,#8]
    add x0,x0,x21,lsl #10
    add x0,x0,x22,lsl #8
    mov x1,x20
    mov x2,#256
    mov x3,x28
    bl d1_dot4_f32_native64
    ldr q0,[x28]
    fmul v0.4s,v0.4s,v9.4s
    ldr x1,[x19,#16]
    add x1,x1,x21,lsl #10
    add x1,x1,x22,lsl #8
    ROUTE_RECORD
    add w22,w22,#1
    cmp w22,#4
    b.lo .Lroute_fine_row
.Lroute_next_group:
    add w21,w21,#1
    cmp w21,#ROUTE_GROUPS
    b.lo .Lroute_mixed_group
    str q10,[x28]
    mov w21,#0
    mov x22,x24
.Lroute_softmax:
    mov x0,x22
    mov w1,w25
    ldr s0,[x28,x21,lsl #2]
    bl lfm_softmax_native
    add x22,x22,#ROUTE_SCORE_STRIDE
    add w21,w21,#1
    cmp w21,#4
    b.lo .Lroute_softmax
    mov x0,x24
    mov x1,x23
    ldr x2,[x19,#48]
    mov w3,w25
    bl d1_route_values4
    add sp,sp,#ROUTE_STACK_PAGES,lsl #12
    LEAVE
.balign 4
.Lroute_log_four: .float 1.3862943611198906
.size d1_route_attention4,.-d1_route_attention4
/* Transpose the raster into quads and keep the mean and radius nearby.
 * x0 and x1 are raster K/V heads. x2 points to the routing job arguments. */
.macro ROUTE_PACK_QUAD source,dest,mean,key=0
    add x3,\source,x10
    add x4,x3,#2,lsl #12
    add x4,x4,#2048 /* second raster row, 40 * 256 */
    ldr q0,[x3]
    ldr q1,[x3,#256]
    ldr q2,[x4]
    ldr q3,[x4,#256]
    add x5,\dest,x10
    str q0,[x5]
    str q1,[x5,#256]
    str q2,[x5,#512]
    str q3,[x5,#768]
    fadd v4.4s,v0.4s,v1.4s
    fadd v4.4s,v4.4s,v2.4s
    fadd v4.4s,v4.4s,v3.4s
    fmul v4.4s,v4.4s,v8.4s
    str q4,[\mean,x10]
    .if \key
    fsub v0.4s,v0.4s,v4.4s
    fsub v1.4s,v1.4s,v4.4s
    fsub v2.4s,v2.4s,v4.4s
    fsub v3.4s,v3.4s,v4.4s
    fmla v16.4s,v0.4s,v0.4s
    fmla v17.4s,v1.4s,v1.4s
    fmla v18.4s,v2.4s,v2.4s
    fmla v19.4s,v3.4s,v3.4s
    .endif
.endm

.balign 64
.global d1_route_pack_head
.type d1_route_pack_head,%function
d1_route_pack_head:
    ENTER
    mov x19,x0
    mov x20,x1
    ldp x21,x22,[x2,#8]
    ldp x23,x24,[x2,#24]
    ldr x25,[x2,#40]
    mov w26,#15
    fmov v8.4s,#0.25
.Lroute_pack_y:
    mov w27,#20
.Lroute_pack_x:
    mov x10,#0
    movi v16.4s,#0
    movi v17.4s,#0
    movi v18.4s,#0
    movi v19.4s,#0
.Lroute_pack_channels:
    ROUTE_PACK_QUAD x19,x21,x23,1
    ROUTE_PACK_QUAD x20,x22,x24
    add x10,x10,#16
    cmp x10,#256
    b.lo .Lroute_pack_channels
    faddp v16.4s,v16.4s,v17.4s
    faddp v18.4s,v18.4s,v19.4s
    faddp v16.4s,v16.4s,v18.4s
    fmaxv s16,v16.4s
    fsqrt s16,s16
    str s16,[x25],#4
    add x21,x21,#1024
    add x22,x22,#1024
    add x23,x23,#256
    add x24,x24,#256
    add x19,x19,#512
    add x20,x20,#512
    subs w27,w27,#1
    b.ne .Lroute_pack_x
    add x19,x19,#2,lsl #12
    add x19,x19,#2048
    add x20,x20,#2,lsl #12
    add x20,x20,#2048
    subs w26,w26,#1
    b.ne .Lroute_pack_y
    LEAVE
.size d1_route_pack_head,.-d1_route_pack_head
.section .note.GNU-stack,"",%progbits
