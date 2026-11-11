/* Adaptive key/value groups on the 40 by 30 post-RoPE vision grid.
 * A 16-byte record indexed by the first position holds a u16 count
 * followed by up to four original positions, all in raster order. */
.arch armv8-a
.include "src/common.inc"
.text

.macro POOL_DOT base
    ldr w0,[sp,x24,lsl #2]
    ldr w1,[sp,x25,lsl #2]
    add x0,\base,x0,lsl #11
    add x1,\base,x1,lsl #11
    mov w2,#512
    bl lfm_dot_native
.endm

.macro POOL_DENOM offset
    add x6,sp,#\offset
    ldr s1,[x6,x24,lsl #2]
    ldr s2,[x6,x25,lsl #2]
    fmul s1,s1,s2
    fcmp s1,#0.0
    b.le .Lpool_zero
.endm

.macro POOL_PAIR a,b,index
    ldr w0,[sp,#(4*\a)]
    ldr w1,[sp,#(4*\b)]
    add x2,x21,x0,lsl #4
    ldr s0,[sp,#(48+4*\index)]
    fcmp s0,s9
    b.lt .Lpool_split_pair\@
    mov w3,#2
    strh w3,[x2]
    strh w0,[x2,#2]
    strh w1,[x2,#4]
    add w23,w23,#1
    b .Lpool_pair_done\@
.Lpool_split_pair\@:
    mov w3,#1
    strh w3,[x2]
    strh w0,[x2,#2]
    add x2,x21,x1,lsl #4
    strh w3,[x2]
    strh w1,[x2,#2]
    add w23,w23,#2
.Lpool_pair_done\@:
.endm

.macro POOL_PRIORITY a,b
    ldr s0,[sp,#(48+4*\a)]
    ldr s1,[sp,#(48+4*\b)]
    fmov s2,#1.0
    fmov s3,wzr
    fadd s4,s0,s2
    fadd s5,s1,s2
    fcmp s0,s9
    fcsel s4,s4,s3,ge
    fcmp s1,s9
    fcsel s5,s5,s3,ge
    fadd s0,s4,s5
.endm

.global d1_vision_pool_plan
.type d1_vision_pool_plan,%function
d1_vision_pool_plan: /* K, V, 1200 records -> group count */
    fmov s0,#0.5
.global d1_vision_pool_plan_cutoff
.type d1_vision_pool_plan_cutoff,%function
d1_vision_pool_plan_cutoff: /* same arrays, s0 cosine cutoff */
    ENTER
    sub sp,sp,#80
    mov x19,x0
    mov x20,x1
    mov x21,x2
    fmov s9,s0
    mov x0,x21
    mov w1,#0
    mov w2,#19200
    bl memset
    mov w22,#0
    mov w23,#0
.Lpool_tile:
    mov w0,#20
    udiv w1,w22,w0
    msub w2,w1,w0,w22
    mov w0,#80
    mul w27,w1,w0
    add w27,w27,w2,lsl #1
    add w1,w27,#1
    add w2,w27,#40
    add w3,w27,#41
    stp w27,w1,[sp]
    stp w2,w3,[sp,#8]
    mov w24,#0
    mov w25,#0
.Lpool_norms:
    POOL_DOT x19
    fsqrt s0,s0
    add x6,sp,#16
    str s0,[x6,x24,lsl #2]
    POOL_DOT x20
    fsqrt s0,s0
    add x6,sp,#32
    str s0,[x6,x24,lsl #2]
    add w24,w24,#1
    mov w25,w24
    cmp w24,#4
    b.lo .Lpool_norms

    mov w24,#0
    mov w26,#0
    fmov s10,#1.0
.Lpool_next_a:
    add w25,w24,#1
.Lpool_compare:
    POOL_DOT x19
    POOL_DENOM 16
    fdiv s8,s0,s1
    POOL_DOT x20
    POOL_DENOM 32
    fdiv s0,s0,s1
    fmin s0,s0,s8
    b .Lpool_store
.Lpool_zero:
    fmov s0,#-1.0
.Lpool_store:
    add x6,sp,#48
    str s0,[x6,x26,lsl #2]
    fmin s10,s10,s0
    add w26,w26,#1
    add w25,w25,#1
    cmp w25,#4
    b.lo .Lpool_compare
    add w24,w24,#1
    cmp w24,#3
    b.lo .Lpool_next_a
    fcmp s10,s9
    b.lt .Lpool_two_pairs

    add x6,x21,x27,lsl #4
    mov w0,#4
    strh w0,[x6]
    ldp w0,w1,[sp]
    ldp w2,w3,[sp,#8]
    strh w0,[x6,#2]
    strh w1,[x6,#4]
    strh w2,[x6,#6]
    strh w3,[x6,#8]
    add w23,w23,#1
    b .Lpool_next_tile
.Lpool_two_pairs:
    POOL_PRIORITY 0,5
    fmov s10,s0
    POOL_PRIORITY 1,4
    fcmp s0,s10
    b.gt .Lpool_vertical
    POOL_PAIR 0,1,0
    POOL_PAIR 2,3,5
    b .Lpool_next_tile
.Lpool_vertical:
    POOL_PAIR 0,2,1
    POOL_PAIR 1,3,4
.Lpool_next_tile:
    add w22,w22,#1
    cmp w22,#300
    b.lo .Lpool_tile
    mov w0,w23
    add sp,sp,#80
    LEAVE
.size d1_vision_pool_plan,.-d1_vision_pool_plan
.size d1_vision_pool_plan_cutoff,.-d1_vision_pool_plan_cutoff

.macro POOL_REDUCE count
    mov w7,#64
.Lpool_vector\@:
    ldp q0,q1,[x9],#32
    .if \count > 1
    ldp q2,q3,[x10],#32
    fadd v0.4s,v0.4s,v2.4s
    fadd v1.4s,v1.4s,v3.4s
    .if \count == 4
    ldp q2,q3,[x11],#32
    fadd v0.4s,v0.4s,v2.4s
    fadd v1.4s,v1.4s,v3.4s
    ldp q2,q3,[x12],#32
    fadd v0.4s,v0.4s,v2.4s
    fadd v1.4s,v1.4s,v3.4s
    .endif
    fmul v0.4s,v0.4s,v4.4s
    fmul v1.4s,v1.4s,v4.4s
    .endif
    stp q0,q1,[x8],#32
    subs w7,w7,#1
    b.ne .Lpool_vector\@
.endm

.global d1_vision_pool_apply
.type d1_vision_pool_apply,%function
d1_vision_pool_apply: /* in-place position rows, records, optional log biases */
    mov w3,#0
    mov w4,#0
    ADDR x13,.Lpool_log
.Lpool_record:
    add x5,x1,x3,lsl #4
    ldrh w6,[x5]
    cbz w6,.Lpool_next_record
    cbz x2,.Lpool_reduce
    ldr s0,[x13,x6,lsl #2]
    str s0,[x2,x4,lsl #2]
.Lpool_reduce:
    add x8,x0,x4,lsl #11
    ldrh w9,[x5,#2]
    add x9,x0,x9,lsl #11
    cmp w6,#1
    b.ne .Lpool_multiple
    cmp x8,x9
    b.eq .Lpool_write_done
    POOL_REDUCE 1
    b .Lpool_write_done
.Lpool_multiple:
    ldrh w10,[x5,#4]
    add x10,x0,x10,lsl #11
    cmp w6,#2
    b.ne .Lpool_four
    fmov v4.4s,#0.5
    POOL_REDUCE 2
    b .Lpool_write_done
.Lpool_four:
    ldrh w11,[x5,#6]
    ldrh w12,[x5,#8]
    add x11,x0,x11,lsl #11
    add x12,x0,x12,lsl #11
    fmov v4.4s,#0.25
    POOL_REDUCE 4
.Lpool_write_done:
    add w4,w4,#1
.Lpool_next_record:
    add w3,w3,#1
    cmp w3,#1200
    b.lo .Lpool_record
    mov w0,w4
    ret
.size d1_vision_pool_apply,.-d1_vision_pool_apply

.section .rodata
.balign 4
.Lpool_log:
    .float 0,0,0.6931471805599453,0,1.3862943611198906
.section .note.GNU-stack,"",%progbits
