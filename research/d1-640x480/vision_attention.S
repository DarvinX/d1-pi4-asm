/* 64-channel attention for the six global-attention vision blocks.
 * Each pass through the value rows accumulates all 64 output channels.
 */
.arch armv8-a
.include "src/common.inc"
.text

#ifndef D1_VALUE_BENCH
.macro D1_VALUE64 a,b,c,d
    ld1 {v0.4s-v3.4s},[x5],#64
    fmla v\a\().4s,v0.4s,v4.s[0]
    fmla v\b\().4s,v1.4s,v4.s[0]
    fmla v\c\().4s,v2.4s,v4.s[0]
    fmla v\d\().4s,v3.4s,v4.s[0]
.endm

.balign 64
.type d1_vision_value64,%function
d1_vision_value64: /* probabilities, value rows, count, stride, output */
    .irp acc,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    movi v\acc\().4s,#0
    .endr
.Lvalue:
    ldr s4,[x0],#4
    mov x5,x1
    D1_VALUE64 16,17,18,19
    D1_VALUE64 20,21,22,23
    D1_VALUE64 24,25,26,27
    D1_VALUE64 28,29,30,31
    add x1,x1,x3
    subs w2,w2,#1
    b.ne .Lvalue
    st1 {v16.4s-v19.4s},[x4],#64
    st1 {v20.4s-v23.4s},[x4],#64
    st1 {v24.4s-v27.4s},[x4],#64
    st1 {v28.4s-v31.4s},[x4]
    ret
.size d1_vision_value64,.-d1_vision_value64

.balign 64
.global d1_vision_attention_head
.type d1_vision_attention_head,%function
d1_vision_attention_head: /* Q, K, V, out, scratch, key count, K/V row stride */
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,x4
    mov w24,w5
    mov x25,x6
    mov w26,#0
    fmov s8,#0.125
    mov w0,#0xff800000
    fmov s9,w0
.Lscore:
    mov x0,x19
    madd x1,x26,x25,x20
    mov w2,#64
    bl lfm_dot_native
    fmul s0,s0,s8
    fmax s9,s9,s0
    str s0,[x23,x26,lsl #2]
    add w26,w26,#1
    cmp w26,w24
    b.lo .Lscore
    mov x0,x23
    mov w1,w24
    fmov s0,s9
    bl lfm_softmax_native
    mov x0,x23
    mov x1,x21
    mov w2,w24
    mov x3,x25
    mov x4,x22
    bl d1_vision_value64
    LEAVE
.size d1_vision_attention_head,.-d1_vision_attention_head
#endif

.macro D1_VALUE_TILE acc0,acc1,acc2,acc3,lane
    fmla v\acc0\().4s,v0.4s,v\lane\().s[0]
    fmla v\acc1\().4s,v1.4s,v\lane\().s[0]
    fmla v\acc2\().4s,v2.4s,v\lane\().s[0]
    fmla v\acc3\().4s,v3.4s,v\lane\().s[0]
.endm

.balign 64
.global d1_vision_value4
.type d1_vision_value4,%function
/* probs, V, out, key count, query stride, V stride, probability stride.
 * Four positions share each load. Consume 16 channels per key sweep. */
d1_vision_value4:
    mov w13,#4
.Lvalue4_channels:
    .irp acc,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    movi v\acc\().4s,#0
    .endr
    mov x9,x0
    add x10,x9,x6
    add x11,x10,x6
    add x12,x11,x6
    mov x7,x1
    mov w8,w3
.Lvalue4_rows:
    ld1 {v0.4s-v3.4s},[x7]
    ldr s4,[x9],#4
    ldr s5,[x10],#4
    ldr s6,[x11],#4
    ldr s7,[x12],#4
    D1_VALUE_TILE 16,17,18,19,4
    D1_VALUE_TILE 20,21,22,23,5
    D1_VALUE_TILE 24,25,26,27,6
    D1_VALUE_TILE 28,29,30,31,7
    add x7,x7,x5
    subs w8,w8,#1
    b.ne .Lvalue4_rows
    mov x7,x2
    st1 {v16.4s-v19.4s},[x7]
    add x7,x7,x4
    st1 {v20.4s-v23.4s},[x7]
    add x7,x7,x4
    st1 {v24.4s-v27.4s},[x7]
    add x7,x7,x4
    st1 {v28.4s-v31.4s},[x7]
    add x1,x1,#64
    add x2,x2,#64
    subs w13,w13,#1
    b.ne .Lvalue4_channels
    ret
.size d1_vision_value4,.-d1_vision_value4

#ifndef D1_VALUE_BENCH
/* One feature per vector, with a lane for each of the four queries. */
.macro D1_PACK_SCORE_QUERY
    mov x9,x19
    add x10,x9,x25
    add x11,x10,x25
    add x12,x11,x25
    add x13,sp,#32
    mov w0,#16
.Lpack_query\@:
    ldr q0,[x9],#16
    ldr q1,[x10],#16
    ldr q2,[x11],#16
    ldr q3,[x12],#16
    trn1 v4.4s,v0.4s,v1.4s
    trn2 v5.4s,v0.4s,v1.4s
    trn1 v6.4s,v2.4s,v3.4s
    trn2 v7.4s,v2.4s,v3.4s
    trn1 v0.2d,v4.2d,v6.2d
    trn1 v1.2d,v5.2d,v7.2d
    trn2 v2.2d,v4.2d,v6.2d
    trn2 v3.2d,v5.2d,v7.2d
    st1 {v0.4s-v3.4s},[x13],#64
    subs w0,w0,#1
    b.ne .Lpack_query\@
.endm

.macro D1_SCORE_GROUP a,b,c,d
    ldr q0,[x0],#16
    ld1 {v1.4s-v4.4s},[x1],#64
    fmla v\a\().4s,v1.4s,v0.s[0]
    fmla v\b\().4s,v2.4s,v0.s[1]
    fmla v\c\().4s,v3.4s,v0.s[2]
    fmla v\d\().4s,v4.4s,v0.s[3]
.endm

.macro D1_SCORE_SUM a,b,c,d
    fadd v\a\().4s,v\a\().4s,v\c\().4s
    fadd v\b\().4s,v\b\().4s,v\d\().4s
    fadd v\a\().4s,v\a\().4s,v\b\().4s
.endm

.macro D1_PACKED_SCORE4
    mov x0,x20
    add x1,sp,#32
    movi v16.4s,#0
    movi v17.4s,#0
    movi v18.4s,#0
    movi v19.4s,#0
    movi v20.4s,#0
    movi v21.4s,#0
    movi v22.4s,#0
    movi v23.4s,#0
    movi v24.4s,#0
    movi v25.4s,#0
    movi v26.4s,#0
    movi v27.4s,#0
    movi v28.4s,#0
    movi v29.4s,#0
    movi v30.4s,#0
    movi v31.4s,#0
    .rept 4
    D1_SCORE_GROUP 16,17,18,19
    D1_SCORE_GROUP 20,21,22,23
    D1_SCORE_GROUP 24,25,26,27
    D1_SCORE_GROUP 28,29,30,31
    .endr
    D1_SCORE_SUM 16,20,24,28
    D1_SCORE_SUM 17,21,25,29
    D1_SCORE_SUM 18,22,26,30
    D1_SCORE_SUM 19,23,27,31
    fadd v0.4s,v16.4s,v17.4s
    fadd v1.4s,v18.4s,v19.4s
    fadd v0.4s,v0.4s,v1.4s
.endm

.macro D1_ATTENTION4 name,packed,pooled=0,values=1
.balign 64
.global \name
.type \name,%function
/* Q, K, V, out, scratch, n_keys, Q/out byte stride, K/V byte stride.
 * Scratch has four rows of ceil(n_keys/4)*4 floats. */
\name:
    ENTER
    sub sp,sp,#32
    .if \packed
    sub sp,sp,#1024
    .endif
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,x4
    mov w24,w5
    mov x25,x6
    add w26,w24,#3
    and w26,w26,#~3
    lsl x26,x26,#2
    mov x27,x7
    .if \pooled
    fmov d10,x8 /* private assembly argument: one log multiplicity per key */
    .endif
    mov w28,#0
    mov w0,#0xff800000
    dup v8.4s,w0
    fmov v9.4s,#0.125
    .if \packed
    D1_PACK_SCORE_QUERY
    .endif
.Lscore4\@:
    .if \packed
    D1_PACKED_SCORE4
    .else
    mov x0,x20
    mov x1,x19
    mov x2,x25
    mov x3,sp
    /* This leaf only touches v0-v4 and v16-v31. */
    bl d1_dot4_f32_native64
    ldr q0,[sp]
    .endif
    fmul v0.4s,v0.4s,v9.4s
    .if \pooled
    fmov x0,d10
    add x0,x0,x28,lsl #2
    ld1r {v1.4s},[x0]
    fadd v0.4s,v0.4s,v1.4s
    .endif
    fmax v8.4s,v8.4s,v0.4s
    add x0,x23,x28,lsl #2
    st1 {v0.s}[0],[x0]
    add x0,x0,x26
    st1 {v0.s}[1],[x0]
    add x0,x0,x26
    st1 {v0.s}[2],[x0]
    add x0,x0,x26
    st1 {v0.s}[3],[x0]
    add x20,x20,x27
    add w28,w28,#1
    cmp w28,w24
    b.lo .Lscore4\@

    str q8,[sp,#16]
    mov x19,x23
    mov w28,#0
.Lsoftmax4\@:
    mov x0,x19
    mov w1,w24
    add x2,sp,#16
    ldr s0,[x2,x28,lsl #2]
    bl lfm_softmax_native
    add x19,x19,x26
    add w28,w28,#1
    cmp w28,#4
    b.lo .Lsoftmax4\@
    .if \values
    mov x0,x23
    mov x1,x21
    mov x2,x22
    mov w3,w24
    mov x4,x25
    mov x5,x27
    mov x6,x26
    bl d1_vision_value4
    .endif
    .if \packed
    add sp,sp,#1024
    .endif
    add sp,sp,#32
    LEAVE
.size \name,.-\name
.endm

D1_ATTENTION4 d1_vision_attention4,0
D1_ATTENTION4 d1_vision_attention4_packed,1
D1_ATTENTION4 d1_vision_attention4_pooled,0,1
D1_ATTENTION4 d1_vision_attention4_probs,0,0,0
#endif
.section .note.GNU-stack,"",%progbits
