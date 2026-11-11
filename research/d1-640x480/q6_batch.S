/* Four-position exact Q6_K × Q8_K tile for d1's down projections.
 * x0 Q6_K row, x1 four Q8_K rows, x2 activation stride in bytes,
 * x3 four FP32 outputs, w4 positive width divisible by 256. */
.arch armv8-a
.text

.macro D1_Q6_ACC ptr,acc,scale0,scale1
    ld1 {v16.16b,v17.16b},[\ptr],#32
    smull v18.8h,v6.8b,v16.8b
    smull v19.8h,v7.8b,v17.8b
    smlal2 v18.8h,v6.16b,v16.16b
    smlal2 v19.8h,v7.16b,v17.16b
    smlal \acc\().4s,v18.4h,v8.h[\scale0]
    smlal2 \acc\().4s,v18.8h,v8.h[\scale0]
    smlal \acc\().4s,v19.4h,v8.h[\scale1]
    smlal2 \acc\().4s,v19.8h,v8.h[\scale1]
.endm

.macro D1_Q6_GROUP qa,qb,shift,nibble,s0,s1
    .if \nibble
    ushr v6.16b,v\qa\().16b,#4
    ushr v7.16b,v\qb\().16b,#4
    .else
    and v6.16b,v\qa\().16b,v25.16b
    and v7.16b,v\qb\().16b,v25.16b
    .endif
    .if \shift
    ushr v28.16b,v4.16b,#\shift
    ushr v29.16b,v5.16b,#\shift
    and v28.16b,v28.16b,v26.16b
    and v29.16b,v29.16b,v26.16b
    .else
    and v28.16b,v4.16b,v26.16b
    and v29.16b,v5.16b,v26.16b
    .endif
    shl v28.16b,v28.16b,#4
    shl v29.16b,v29.16b,#4
    orr v6.16b,v6.16b,v28.16b
    orr v7.16b,v7.16b,v29.16b
    sub v6.16b,v6.16b,v27.16b
    sub v7.16b,v7.16b,v27.16b
    D1_Q6_ACC x9,v10,\s0,\s1
    D1_Q6_ACC x10,v11,\s0,\s1
    D1_Q6_ACC x11,v12,\s0,\s1
    D1_Q6_ACC x12,v13,\s0,\s1
.endm

.macro D1_Q6_HALF lo,hi,scale
    ldp q0,q1,[x0,#\lo]
    ldp q2,q3,[x0,#(\lo)+32]
    ldp q4,q5,[x0,#\hi]
    ldr d8,[x0,#\scale]
    sxtl v8.8h,v8.8b
    D1_Q6_GROUP 0,1,0,0,0,1
    D1_Q6_GROUP 2,3,2,0,2,3
    D1_Q6_GROUP 0,1,4,1,4,5
    D1_Q6_GROUP 2,3,6,1,6,7
.endm

.macro D1_Q6_FOLD dst,acc,act
    addv s18,\acc\().4s
    scvtf s18,s18
    ldr s17,[\act]
    fmul s17,s24,s17
    fmadd \dst,s17,s18,\dst
.endm

.macro D1_Q6_EXPANDED_GROUP offset,s0,s1
    ldp q6,q7,[x0,#\offset]
    D1_Q6_ACC x9,v10,\s0,\s1
    D1_Q6_ACC x10,v11,\s0,\s1
    D1_Q6_ACC x11,v12,\s0,\s1
    D1_Q6_ACC x12,v13,\s0,\s1
.endm

.macro D1_Q6_EXPANDED_HALF offset,scales
    ldr d8,[x0,#\scales]
    sxtl v8.8h,v8.8b
    D1_Q6_EXPANDED_GROUP \offset,0,1
    D1_Q6_EXPANDED_GROUP (\offset+32),2,3
    D1_Q6_EXPANDED_GROUP (\offset+64),4,5
    D1_Q6_EXPANDED_GROUP (\offset+96),6,7
.endm

.macro D1_Q6_TILE name,expanded
.balign 64
.global \name
.type \name,%function
\name:
    stp d10,d11,[sp,#-48]!
    stp d12,d13,[sp,#16]
    str d8,[sp,#32]
    add x5,x1,x2
    add x6,x5,x2
    add x7,x6,x2
    fmov s20,wzr
    fmov s21,wzr
    fmov s22,wzr
    fmov s23,wzr
    .if !\expanded
    movi v25.16b,#15
    movi v26.16b,#3
    movi v27.16b,#32
    .endif
1:
    .if \expanded
    ldr h24,[x0,#272]
    .else
    ldr h24,[x0,#208]
    .endif
    fcvt s24,h24
    movi v10.4s,#0
    movi v11.4s,#0
    movi v12.4s,#0
    movi v13.4s,#0
    add x9,x1,#32
    add x10,x5,#32
    add x11,x6,#32
    add x12,x7,#32
    .if \expanded
    D1_Q6_EXPANDED_HALF 0,256
    D1_Q6_EXPANDED_HALF 128,264
    .else
    D1_Q6_HALF 0,128,192
    D1_Q6_HALF 64,160,200
    .endif
    D1_Q6_FOLD s20,v10,x1
    D1_Q6_FOLD s21,v11,x5
    D1_Q6_FOLD s22,v12,x6
    D1_Q6_FOLD s23,v13,x7
    .if \expanded
    add x0,x0,#288
    .else
    add x0,x0,#210
    .endif
    add x1,x1,#320
    add x5,x5,#320
    add x6,x6,#320
    add x7,x7,#320
    subs w4,w4,#256
    b.ne 1b
    stp s20,s21,[x3]
    stp s22,s23,[x3,#8]
    ldr d8,[sp,#32]
    ldp d12,d13,[sp,#16]
    ldp d10,d11,[sp],#48
    ret
.size \name,.-\name
.endm

D1_Q6_TILE d1_dot4_q6_k,0
#ifdef D1_BENCH_Q6_EXPANDED
D1_Q6_TILE d1_dot4_q6_k_expanded,1
#endif
.section .note.GNU-stack,"",%progbits
