/* Four Q8_K activations share one stream of packed Q4_K weights.
 * x0: row, x1: packed activation 0, x2: activation stride in bytes,
 * x3: four FP32 results, w4: input width (positive multiple of 256).
 * Use the same FP32 operations in the same order as H42 for each output.
 */
.arch armv8-a
.text

.macro D1_MIN act,acc,scale
    ldr q4,[\act,#16]
    smull v6.4s,v4.4h,v9.4h
    smlal2 v6.4s,v4.8h,v9.8h
    addv s6,v6.4s
    scvtf s6,s6
    ldr \scale,[\act]
    fmul s7,s17,\scale
    fmsub \acc,s7,s6,\acc
.endm

.macro D1_ACC ptr,acc,g
    ld1 {v4.16b,v5.16b},[\ptr],#32
    smull v6.8h,v2.8b,v4.8b
    smlal2 v6.8h,v2.16b,v4.16b
    smull v7.8h,v3.8b,v5.8b
    smlal2 v7.8h,v3.16b,v5.16b
    add v6.8h,v6.8h,v7.8h
    smlal \acc\().4s,v6.4h,v8.h[\g]
    smlal2 \acc\().4s,v6.8h,v8.h[\g]
.endm

.macro D1_GROUP hi,g
    .if \hi
    ushr v2.16b,v0.16b,#4
    ushr v3.16b,v1.16b,#4
    .else
    and v2.16b,v0.16b,v27.16b
    and v3.16b,v1.16b,v27.16b
    .endif
    D1_ACC x9,v10,\g
    D1_ACC x10,v11,\g
    D1_ACC x11,v12,\g
    D1_ACC x12,v13,\g
.endm

/* Keep four multiply-accumulate chains live so the A72 can issue the
 * next position while the previous product is in flight. Each lane
 * sums only four byte products (maximum magnitude 7680). */
.macro D1_GROUP_ILP hi,g,across=0
    .if \hi
    ushr v2.16b,v0.16b,#4
    ushr v3.16b,v1.16b,#4
    .else
    and v2.16b,v0.16b,v27.16b
    and v3.16b,v1.16b,v27.16b
    .endif
    ld1 {v14.16b,v15.16b},[x9],#32
    ld1 {v18.16b,v19.16b},[x10],#32
    ld1 {v24.16b,v25.16b},[x11],#32
    ld1 {v28.16b,v29.16b},[x12],#32
    smull v4.8h,v2.8b,v14.8b
    smull v5.8h,v2.8b,v18.8b
    smull v6.8h,v2.8b,v24.8b
    smull v7.8h,v2.8b,v28.8b
    smlal2 v4.8h,v2.16b,v14.16b
    smlal2 v5.8h,v2.16b,v18.16b
    smlal2 v6.8h,v2.16b,v24.16b
    smlal2 v7.8h,v2.16b,v28.16b
    smlal v4.8h,v3.8b,v15.8b
    smlal v5.8h,v3.8b,v19.8b
    smlal v6.8h,v3.8b,v25.8b
    smlal v7.8h,v3.8b,v29.8b
    smlal2 v4.8h,v3.16b,v15.16b
    smlal2 v5.8h,v3.16b,v19.16b
    smlal2 v6.8h,v3.16b,v25.16b
    smlal2 v7.8h,v3.16b,v29.16b
    .if \across
    /* Each 16-bit lane holds four products. Widen before the
       horizontal sum of the 32-channel group to avoid overflow. */
    saddlp v4.4s,v4.8h
    saddlp v5.4s,v5.8h
    saddlp v6.4s,v6.8h
    saddlp v7.4s,v7.8h
    addp v4.4s,v4.4s,v5.4s
    addp v6.4s,v6.4s,v7.4s
    addp v4.4s,v4.4s,v6.4s
    .if \g < 4
    mla v10.4s,v4.4s,v8.s[\g]
    .else
    mla v10.4s,v4.4s,v9.s[\g-4]
    .endif
    .else
    smlal v10.4s,v4.4h,v8.h[\g]
    smlal v11.4s,v5.4h,v8.h[\g]
    smlal v12.4s,v6.4h,v8.h[\g]
    smlal v13.4s,v7.4h,v8.h[\g]
    smlal2 v10.4s,v4.8h,v8.h[\g]
    smlal2 v11.4s,v5.8h,v8.h[\g]
    smlal2 v12.4s,v6.8h,v8.h[\g]
    smlal2 v13.4s,v7.8h,v8.h[\g]
    .endif
.endm

.macro D1_GROUP_ACROSS hi,g
    D1_GROUP_ILP \hi,\g,1
.endm

.macro D1_MIN_ACROSS act,lane
    ldr q4,[\act,#16]
    smull v6.4s,v4.4h,v9.4h
    smlal2 v6.4s,v4.8h,v9.8h
    addv s6,v6.4s
    mov v11.s[\lane],v6.s[0]
    ld1 {v30.s}[\lane],[\act]
.endm

.macro D1_FOLD acc,vsum,scale
    addv s6,\vsum\().4s
    scvtf s6,s6
    fmul s7,s16,\scale
    fmadd \acc,s7,s6,\acc
.endm

.macro D1_Q4_TILE name,group,ilp,across=0
.balign 64
.global \name
.type \name,%function
\name:
    .if \ilp
    stp d14,d15,[sp,#-16]!
    .endif
    stp d8,d9,[sp,#-48]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    add x5,x1,x2
    add x6,x5,x2
    add x7,x6,x2
    .if \across
    movi v20.4s,#0
    .else
    fmov s20,wzr
    fmov s21,wzr
    fmov s22,wzr
    fmov s23,wzr
    .endif
    movi v27.16b,#15
1:
    ldr h16,[x0]
    ldr h17,[x0,#2]
    fcvt s16,h16
    fcvt s17,h17
    ldp w9,w10,[x0,#4]
    ldr w11,[x0,#12]
    and w12,w9,#0x3f3f3f3f
    and w13,w10,#0x3f3f3f3f
    lsr w14,w9,#2
    lsr w15,w10,#2
    and w14,w14,#0x30303030
    and w15,w15,#0x30303030
    and w16,w11,#0x0f0f0f0f
    lsr w17,w11,#4
    and w17,w17,#0x0f0f0f0f
    orr w14,w14,w16
    orr w15,w15,w17
    orr x12,x12,x14,lsl #32
    orr x13,x13,x15,lsl #32
    fmov d8,x12
    fmov d9,x13
    uxtl v8.8h,v8.8b
    uxtl v9.8h,v9.8b
    .if \across
    D1_MIN_ACROSS x1,0
    D1_MIN_ACROSS x5,1
    D1_MIN_ACROSS x6,2
    D1_MIN_ACROSS x7,3
    scvtf v11.4s,v11.4s
    fmul v7.4s,v30.4s,v17.s[0]
    fmls v20.4s,v7.4s,v11.4s
    uxtl2 v9.4s,v8.8h
    uxtl v8.4s,v8.4h
    movi v10.4s,#0
    .else
    D1_MIN x1,s20,s24
    D1_MIN x5,s21,s25
    D1_MIN x6,s22,s26
    D1_MIN x7,s23,s28
    movi v10.4s,#0
    movi v11.4s,#0
    movi v12.4s,#0
    movi v13.4s,#0
    .endif
    add x8,x0,#16
    add x9,x1,#32
    add x10,x5,#32
    add x11,x6,#32
    add x12,x7,#32
    ld1 {v0.16b,v1.16b},[x8],#32
    \group 0,0
    \group 1,1
    ld1 {v0.16b,v1.16b},[x8],#32
    \group 0,2
    \group 1,3
    ld1 {v0.16b,v1.16b},[x8],#32
    \group 0,4
    \group 1,5
    ld1 {v0.16b,v1.16b},[x8],#32
    \group 0,6
    \group 1,7
    .if \across
    scvtf v10.4s,v10.4s
    fmul v7.4s,v30.4s,v16.s[0]
    fmla v20.4s,v7.4s,v10.4s
    .else
    .if \ilp
    ldr s24,[x1]
    ldr s25,[x5]
    ldr s28,[x7]
    .endif
    D1_FOLD s20,v10,s24
    D1_FOLD s21,v11,s25
    D1_FOLD s22,v12,s26
    D1_FOLD s23,v13,s28
    .endif
    add x0,x0,#144
    add x1,x1,#320
    add x5,x5,#320
    add x6,x6,#320
    add x7,x7,#320
    subs w4,w4,#256
    b.ne 1b
    .if \across
    str q20,[x3]
    .else
    stp s20,s21,[x3]
    stp s22,s23,[x3,#8]
    .endif
    ldp d12,d13,[sp,#32]
    ldp d10,d11,[sp,#16]
    ldp d8,d9,[sp],#48
    .if \ilp
    ldp d14,d15,[sp],#16
    .endif
    ret
.size \name,.-\name
.endm
D1_Q4_TILE d1_dot4_q4_k,D1_GROUP,0
D1_Q4_TILE d1_dot4_q4_k_ilp,D1_GROUP_ILP,1
#if defined(D1_BENCH_Q4_ACROSS) || defined(D1_BENCH_Q4_WIDE) || defined(D1_BATCH)
D1_Q4_TILE d1_dot4_q4_k_across,D1_GROUP_ACROSS,1,1
#endif
.section .note.GNU-stack,"",%progbits
