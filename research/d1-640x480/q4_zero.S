/* Q4_0 × Q8_0 on Cortex-A72 without DotProd. Four signed products fit
 * in an int16 lane before the pairwise widening reduction. Both leaves
 * preserve the pinned Arm fallback's even/odd FP32 accumulation order. */
.arch armv8-a
.text

.macro D1_Q40_WEIGHT
    ldr h23,[x0],#2
    fcvt s23,h23
    ldr q0,[x0],#16
    and v1.16b,v0.16b,v31.16b
    ushr v0.16b,v0.16b,#4
    sub v1.16b,v1.16b,v22.16b
    sub v0.16b,v0.16b,v22.16b
.endm
.macro D1_Q40_ACT input,acc
    ldr h24,[\input],#2
    fcvt s24,h24
    fmul s24,s24,s23
    ld1 {v2.16b,v3.16b},[\input],#32
    smull v4.8h,v1.8b,v2.8b
    smlal2 v4.8h,v1.16b,v2.16b
    smlal v4.8h,v0.8b,v3.8b
    smlal2 v4.8h,v0.16b,v3.16b
    saddlp v5.4s,v4.8h
    scvtf v5.4s,v5.4s
    fmla v\acc\().4s,v5.4s,v24.s[0]
.endm
.macro D1_Q40_FOLD even,odd
    faddp v\even\().4s,v\even\().4s,v\even\().4s
    faddp v\odd\().4s,v\odd\().4s,v\odd\().4s
    faddp s\even,v\even\().2s
    faddp s\odd,v\odd\().2s
    fadd s0,s\even,s\odd
.endm

.balign 64
.global d1_dot_q4_0
.type d1_dot_q4_0,%function
d1_dot_q4_0: /* Q4_0 row, Q8_0 row, width (positive multiple of 64) */
    movi v31.16b,#15
    movi v22.16b,#8
    movi v20.4s,#0
    movi v21.4s,#0
1:  D1_Q40_WEIGHT
    D1_Q40_ACT x1,20
    D1_Q40_WEIGHT
    D1_Q40_ACT x1,21
    subs w2,w2,#64
    b.ne 1b
    D1_Q40_FOLD 20,21
    ret
.size d1_dot_q4_0,.-d1_dot_q4_0

.balign 64
.global d1_dot4_q4_0
.type d1_dot4_q4_0,%function
d1_dot4_q4_0: /* one Q4_0 row, four Q8_0 rows, stride, out[4], width */
    add x5,x1,x2
    add x6,x5,x2
    add x7,x6,x2
    movi v31.16b,#15
    movi v22.16b,#8
    .irp reg,20,21,25,26,27,28,29,30
    movi v\reg\().4s,#0
    .endr
1:  D1_Q40_WEIGHT
    D1_Q40_ACT x1,20
    D1_Q40_ACT x5,25
    D1_Q40_ACT x6,27
    D1_Q40_ACT x7,29
    D1_Q40_WEIGHT
    D1_Q40_ACT x1,21
    D1_Q40_ACT x5,26
    D1_Q40_ACT x6,28
    D1_Q40_ACT x7,30
    subs w4,w4,#64
    b.ne 1b
    D1_Q40_FOLD 20,21
    str s0,[x3]
    D1_Q40_FOLD 25,26
    str s0,[x3,#4]
    D1_Q40_FOLD 27,28
    str s0,[x3,#8]
    D1_Q40_FOLD 29,30
    str s0,[x3,#12]
    ret
.size d1_dot4_q4_0,.-d1_dot4_q4_0
.section .note.GNU-stack,"",%progbits
