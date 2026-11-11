/* Preserve the original Q8_0 ILP partial sums and even/odd reductions.
 * Prepack two weight rows and four activations so each scale is
 * widened once and one SIMD multiply scales all four positions. */
.arch armv8-a
.text

.global d1_q8_ilp_pack_a4
.type d1_q8_ilp_pack_a4,%function
/* x0 raw Q8_0 rows, x1 packed output, x2 raw row byte stride,
 * w3 positive group count, w4 positive four-position tile count. */
d1_q8_ilp_pack_a4:
.Lilp_pack_tile:
    mov x8,x0
    add x9,x8,x2
    add x10,x9,x2
    add x11,x10,x2
    mov w12,w3
.Lilp_pack_group:
    ldr h0,[x8],#2
    ldr h1,[x9],#2
    ldr h2,[x10],#2
    ldr h3,[x11],#2
    fcvt s0,h0
    fcvt s1,h1
    fcvt s2,h2
    fcvt s3,h3
    mov v0.s[1],v1.s[0]
    mov v0.s[2],v2.s[0]
    mov v0.s[3],v3.s[0]
    str q0,[x1],#16
    ldp q0,q1,[x8],#32
    stp q0,q1,[x1],#32
    ldp q0,q1,[x9],#32
    stp q0,q1,[x1],#32
    ldp q0,q1,[x10],#32
    stp q0,q1,[x1],#32
    ldp q0,q1,[x11],#32
    stp q0,q1,[x1],#32
    subs w12,w12,#1
    b.ne .Lilp_pack_group
    add x0,x0,x2,lsl #2
    subs w4,w4,#1
    b.ne .Lilp_pack_tile
    ret
.size d1_q8_ilp_pack_a4,.-d1_q8_ilp_pack_a4

.macro ILP_PACKED_DOT lane,left,right
    ldp q4,q5,[x0],#32
    smull v6.8h,v0.8b,v4.8b
    smull v7.8h,v1.8b,v5.8b
    smull v8.8h,v2.8b,v4.8b
    smull v9.8h,v3.8b,v5.8b
    smlal2 v6.8h,v0.16b,v4.16b
    smlal2 v7.8h,v1.16b,v5.16b
    smlal2 v8.8h,v2.16b,v4.16b
    smlal2 v9.8h,v3.16b,v5.16b
    saddlp v10.4s,v6.8h
    saddlp v11.4s,v8.8h
    sadalp v10.4s,v7.8h
    sadalp v11.4s,v9.8h
    scvtf v10.4s,v10.4s
    scvtf v11.4s,v11.4s
    fmla v\left\().4s,v10.4s,v12.s[\lane]
    fmla v\right\().4s,v11.4s,v13.s[\lane]
.endm

.macro ILP_PACKED_GROUP a,b,c,d,e,f,g,h
    ldr d10,[x1],#8
    ldp q0,q1,[x1],#32
    ldp q2,q3,[x1],#32
    ldr q12,[x0],#16
    fmul v13.4s,v12.4s,v10.s[1]
    fmul v12.4s,v12.4s,v10.s[0]
    ILP_PACKED_DOT 0,\a,\e
    ILP_PACKED_DOT 1,\b,\f
    ILP_PACKED_DOT 2,\c,\g
    ILP_PACKED_DOT 3,\d,\h
.endm

.macro ILP_PACKED_SUM left_even,left_odd,right_even,right_odd
    faddp v0.4s,v\left_even\().4s,v\left_odd\().4s
    faddp v1.4s,v\right_even\().4s,v\right_odd\().4s
    faddp v0.4s,v0.4s,v1.4s
    faddp v0.4s,v0.4s,v0.4s
    str d0,[x2]
    add x2,x2,x3
.endm

.balign 64
.global d1_q8_ilp_4x2
.type d1_q8_ilp_4x2,%function
/* x0 packed four positions, x1 packed two weights, x2 output,
 * x3 output row stride in bytes, w4 positive count of 32-channel groups,
 * divisible by two. */
d1_q8_ilp_4x2:
    stp d8,d9,[sp,#-48]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    .irp acc,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
        movi v\acc\().4s,#0
    .endr
.Lilp_packed_group:
    ILP_PACKED_GROUP 16,18,20,22,24,26,28,30
    ILP_PACKED_GROUP 17,19,21,23,25,27,29,31
    subs w4,w4,#2
    b.ne .Lilp_packed_group
    ILP_PACKED_SUM 16,17,24,25
    ILP_PACKED_SUM 18,19,26,27
    ILP_PACKED_SUM 20,21,28,29
    ILP_PACKED_SUM 22,23,30,31
    ldp d12,d13,[sp,#32]
    ldp d10,d11,[sp,#16]
    ldp d8,d9,[sp],#48
    ret
.size d1_q8_ilp_4x2,.-d1_q8_ilp_4x2
.section .note.GNU-stack,"",%progbits
