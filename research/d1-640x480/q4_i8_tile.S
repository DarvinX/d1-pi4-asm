/* Exact 4-position × 16-output Q4_K tile. A keeps the original Q8_K
 * scale, group sums and codes. B keeps four-bit codes and widens only
 * the group metadata. All offsets below are per 256-channel block. */
.arch armv8-a
.text

.balign 32
.global d1_q4_i8_pack_a4
.type d1_q4_i8_pack_a4,%function
/* x0 first Q8_K block, x1 packed tile, x2 original row stride. */
d1_q4_i8_pack_a4:
    add x3,x0,x2
    add x4,x3,x2
    add x5,x4,x2
    ld1 {v0.s}[0],[x0]
    ld1 {v0.s}[1],[x3]
    ld1 {v0.s}[2],[x4]
    ld1 {v0.s}[3],[x5]
    str q0,[x1],#16
    ldr q1,[x0,#16]
    ldr q2,[x3,#16]
    ldr q3,[x4,#16]
    ldr q4,[x5,#16]
    zip1 v5.8h,v1.8h,v2.8h
    zip2 v6.8h,v1.8h,v2.8h
    zip1 v7.8h,v3.8h,v4.8h
    zip2 v16.8h,v3.8h,v4.8h
    zip1 v0.4s,v5.4s,v7.4s
    zip2 v1.4s,v5.4s,v7.4s
    zip1 v2.4s,v6.4s,v16.4s
    zip2 v3.4s,v6.4s,v16.4s
    stp q0,q1,[x1],#32
    stp q2,q3,[x1],#32
    add x0,x0,#32
    add x3,x3,#32
    add x4,x4,#32
    add x5,x5,#32
    mov w6,#16
.Lq4_i8_pack_codes:
    ldr q0,[x0],#16
    ldr q1,[x3],#16
    ldr q2,[x4],#16
    ldr q3,[x5],#16
    zip1 v4.16b,v0.16b,v1.16b
    zip2 v5.16b,v0.16b,v1.16b
    zip1 v6.16b,v2.16b,v3.16b
    zip2 v7.16b,v2.16b,v3.16b
    zip1 v0.8h,v4.8h,v6.8h
    zip2 v1.8h,v4.8h,v6.8h
    zip1 v2.8h,v5.8h,v7.8h
    zip2 v3.8h,v5.8h,v7.8h
    stp q0,q1,[x1],#32
    stp q2,q3,[x1],#32
    subs w6,w6,#1
    b.ne .Lq4_i8_pack_codes
    ret
.size d1_q4_i8_pack_a4,.-d1_q4_i8_pack_a4

.macro q4_i8_mac lane,lo,hi,seed
    dup v3.8b,v2.b[\lane]
    .if \seed
    smull v\lo\().8h,v1.8b,v3.8b
    smull v\hi\().8h,v0.8b,v3.8b
    .else
    smlal v\lo\().8h,v1.8b,v3.8b
    smlal v\hi\().8h,v0.8b,v3.8b
    .endif
.endm

.macro q4_i8_step seed=0
    ldr d0,[x6],#8
    ldr s2,[x5],#4
    and v1.8b,v0.8b,v7.8b
    ushr v0.8b,v0.8b,#4
    q4_i8_mac 0,8,9,\seed
    q4_i8_mac 1,10,11,\seed
    q4_i8_mac 2,12,13,\seed
    q4_i8_mac 3,14,15,\seed
.endm

.macro q4_i8_scale partial,lo,hi,scale
    smlal v\lo\().4s,v\partial\().4h,v\scale\().4h
    smlal2 v\hi\().4s,v\partial\().8h,v\scale\().8h
.endm

.macro q4_i8_write offset,dot,minimum
    ldr d0,[x16],#8
    ldr d1,[x17],#8
    fcvtl v0.4s,v0.4h
    fcvtl v1.4s,v1.4h
    fmul v0.4s,v0.4s,v4.s[0]
    fmul v1.4s,v1.4s,v4.s[0]
    scvtf v\dot\().4s,v\dot\().4s
    scvtf v\minimum\().4s,v\minimum\().4s
    ldr q3,[x15,#\offset]
    fmls v3.4s,v1.4s,v\minimum\().4s
    fmla v3.4s,v0.4s,v\dot\().4s
    str q3,[x15,#\offset]
.endm

.macro q4_i8_store lane,d0,d1,d2,d3,m0,m1,m2,m3
    ldr s4,[x0,#\lane*4]
    mov x16,x1
    add x17,x1,#32
    q4_i8_write 0,\d0,\m0
    q4_i8_write 16,\d1,\m1
    q4_i8_write 32,\d2,\m2
    q4_i8_write 48,\d3,\m3
    add x15,x15,x3
.endm

.macro q4_i8_finish lane,d0,d1,d2,d3
    movi v8.4s,#0
    movi v9.4s,#0
    movi v10.4s,#0
    movi v11.4s,#0
    add x13,x1,#192
    add x14,x0,#16+\lane*2
    mov w12,#8
.Lq4_i8_min\@:
    ldr q0,[x13],#16
    ldr h3,[x14],#8
    uxtl v1.8h,v0.8b
    uxtl2 v2.8h,v0.16b
    smlal v8.4s,v1.4h,v3.h[0]
    smlal2 v9.4s,v1.8h,v3.h[0]
    smlal v10.4s,v2.4h,v3.h[0]
    smlal2 v11.4s,v2.8h,v3.h[0]
    subs w12,w12,#1
    b.ne .Lq4_i8_min\@
    q4_i8_store \lane,\d0,\d1,\d2,\d3,8,9,10,11
.endm

.macro q4_i8_finish_pair first,second,d0,d1,d2,d3,d4,d5,d6,d7
    movi v8.4s,#0
    movi v9.4s,#0
    movi v10.4s,#0
    movi v11.4s,#0
    movi v12.4s,#0
    movi v13.4s,#0
    movi v14.4s,#0
    movi v15.4s,#0
    add x13,x1,#192
    add x14,x0,#16+\first*2
    mov w12,#8
.Lq4_i8_min_pair\@:
    ldr q0,[x13],#16
    ldr s3,[x14],#8
    uxtl v1.8h,v0.8b
    uxtl2 v2.8h,v0.16b
    smlal v8.4s,v1.4h,v3.h[0]
    smlal2 v9.4s,v1.8h,v3.h[0]
    smlal v10.4s,v2.4h,v3.h[0]
    smlal2 v11.4s,v2.8h,v3.h[0]
    smlal v12.4s,v1.4h,v3.h[1]
    smlal2 v13.4s,v1.8h,v3.h[1]
    smlal v14.4s,v2.4h,v3.h[1]
    smlal2 v15.4s,v2.8h,v3.h[1]
    subs w12,w12,#1
    b.ne .Lq4_i8_min_pair\@
    q4_i8_store \first,\d0,\d1,\d2,\d3,8,9,10,11
    q4_i8_store \second,\d4,\d5,\d6,\d7,12,13,14,15
.endm

.macro Q4_I8_TILE name,guarded=0,seed=0,min_pair=0
.balign 64
.global \name
.type \name,%function
/* x0 packed activations, x1 packed weights, x2 output, x3 output
 * row stride in bytes, w4 positive number of 256-channel blocks.
 * The guarded variant takes a mask in w5, one bit per block. */
\name:
    .if \guarded
    str x19,[sp,#-16]!
    mov w19,w5
    .endif
    stp d8,d9,[sp,#-64]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    stp d14,d15,[sp,#48]
    movi v0.4s,#0
    mov x5,x2
    mov w7,#4
.Lq4_i8_init\@:
    stp q0,q0,[x5]
    stp q0,q0,[x5,#32]
    add x5,x5,x3
    subs w7,w7,#1
    b.ne .Lq4_i8_init\@
    movi v7.8b,#15
.Lq4_i8_block\@:
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
    add x5,x0,#80
    add x6,x1,#320
    add x8,x1,#64
    mov w9,#8
    .if \guarded
    mov w7,#8
    mov w12,#2
    tbz w19,#0,.Lq4_i8_length\@
    mov w7,#16
    mov w12,#1
.Lq4_i8_length\@:
    lsr w19,w19,#1
    .endif
.Lq4_i8_group\@:
    ldr q0,[x8],#16
    uxtl v5.8h,v0.8b
    uxtl2 v6.8h,v0.16b
    .if \guarded
    mov w10,w12
    .else
    mov w10,#2
    .endif
.Lq4_i8_half\@:
    .if !\seed
    movi v8.8h,#0
    movi v9.8h,#0
    movi v10.8h,#0
    movi v11.8h,#0
    movi v12.8h,#0
    movi v13.8h,#0
    movi v14.8h,#0
    movi v15.8h,#0
    .endif
    .if \guarded
    mov w11,w7
    .else
    mov w11,#8
    .endif
    .if \seed
    q4_i8_step 1
    q4_i8_step
    sub w11,w11,#1
    .endif
.Lq4_i8_inner\@:
    q4_i8_step
    q4_i8_step
    subs w11,w11,#1
    b.ne .Lq4_i8_inner\@
    /* Unchecked groups widen at sixteen terms. A certified group
     * has a bound on every prefix of the full thirty-two-term dot. */
    q4_i8_scale 8,16,17,5
    q4_i8_scale 9,18,19,6
    q4_i8_scale 10,20,21,5
    q4_i8_scale 11,22,23,6
    q4_i8_scale 12,24,25,5
    q4_i8_scale 13,26,27,6
    q4_i8_scale 14,28,29,5
    q4_i8_scale 15,30,31,6
    subs w10,w10,#1
    b.ne .Lq4_i8_half\@
    subs w9,w9,#1
    b.ne .Lq4_i8_group\@
    mov x15,x2
    .if \min_pair
    q4_i8_finish_pair 0,1,16,17,18,19,20,21,22,23
    q4_i8_finish_pair 2,3,24,25,26,27,28,29,30,31
    .else
    q4_i8_finish 0,16,17,18,19
    q4_i8_finish 1,20,21,22,23
    q4_i8_finish 2,24,25,26,27
    q4_i8_finish 3,28,29,30,31
    .endif
    add x0,x0,#1104
    add x1,x1,#2368
    subs w4,w4,#1
    b.ne .Lq4_i8_block\@
    ldp d10,d11,[sp,#16]
    ldp d12,d13,[sp,#32]
    ldp d14,d15,[sp,#48]
    ldp d8,d9,[sp],#64
    .if \guarded
    ldr x19,[sp],#16
    .endif
    ret
.size \name,.-\name
.endm

Q4_I8_TILE d1_q4_i8_exact_4x16

.ifdef D1_Q4_SEED_PRODUCT
Q4_I8_TILE d1_q4_i8_seeded_4x16,0,1
.endif
.ifdef D1_Q4_MIN_PAIR
Q4_I8_TILE d1_q4_i8_min_pair_4x16,0,0,1
.endif

.ifdef D1_Q4_GUARD32
Q4_I8_TILE d1_q4_i8_guarded_4x16,1

.macro q4_guard_lane lo,hi
    abs v\lo\().16b,v\lo\().16b
    abs v\hi\().16b,v\hi\().16b
    /* Treat abs(-128)'s byte 0x80 as unsigned 128. */
    uaddlp v16.8h,v\lo\().16b
    uadalp v16.8h,v\hi\().16b
    uaddlv s16,v16.8h
    fmov w7,s16
    cmp w7,#2184
    b.hi .Lguard32_advance
.endm

.balign 32
.global d1_q4_i8_guard32
.type d1_q4_i8_guard32,%function
/* x0 packed activation tile, w1 number of blocks (at most 32).
 * Bit i certifies all groups and all four positions in block i. */
d1_q4_i8_guard32:
    mov w2,#0
    mov w3,#1
    cbz w1,.Lguard32_done
.Lguard32_block:
    add x5,x0,#80
    mov w6,#8
.Lguard32_group:
    ld4 {v0.16b-v3.16b},[x5],#64
    ld4 {v4.16b-v7.16b},[x5],#64
    q4_guard_lane 0,4
    q4_guard_lane 1,5
    q4_guard_lane 2,6
    q4_guard_lane 3,7
    subs w6,w6,#1
    b.ne .Lguard32_group
    orr w2,w2,w3
.Lguard32_advance:
    add x0,x0,#1104
    lsl w3,w3,#1
    subs w1,w1,#1
    b.ne .Lguard32_block
.Lguard32_done:
    mov w0,w2
    ret
.size d1_q4_i8_guard32,.-d1_q4_i8_guard32
.endif
.section .note.GNU-stack,"",%progbits
