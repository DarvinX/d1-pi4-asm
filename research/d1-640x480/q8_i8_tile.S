/* A 4-position × 16-output Q8_0 tile. Two channels share a broadcast.
 * Integer dot products are exact and each 32-channel group scales once. */
.arch armv8-a
.text

.macro q8_i8_mac lane,a,b,c,d
    dup v3.8h,v2.h[\lane]
    smull v8.8h,v0.8b,v3.8b
    smull2 v9.8h,v0.16b,v3.16b
    smull v10.8h,v1.8b,v3.8b
    smull2 v11.8h,v1.16b,v3.16b
    sadalp v\a\().4s,v8.8h
    sadalp v\b\().4s,v9.8h
    sadalp v\c\().4s,v10.8h
    sadalp v\d\().4s,v11.8h
.endm

.macro q8_i8_pair
    ldp q0,q1,[x6],#32
    ldr d2,[x5],#8
    q8_i8_mac 0,16,17,18,19
    q8_i8_mac 1,20,21,22,23
    q8_i8_mac 2,24,25,26,27
    q8_i8_mac 3,28,29,30,31
.endm

.macro q8_i8_scale offset,dot
    ldr d0,[x9],#8
    fcvtl v0.4s,v0.4h
    fmul v0.4s,v0.4s,v4.s[0]
    scvtf v\dot\().4s,v\dot\().4s
    ldr q1,[x8,#\offset]
    fmla v1.4s,v0.4s,v\dot\().4s
    str q1,[x8,#\offset]
.endm

.macro q8_i8_finish lane,a,b,c,d
    ldr h4,[x0,#2*\lane]
    fcvt s4,h4
    mov x9,x1
    q8_i8_scale 0,\a
    q8_i8_scale 16,\b
    q8_i8_scale 32,\c
    q8_i8_scale 48,\d
    add x8,x8,x3
.endm

.balign 64
.global d1_q8_i8_4x16
.type d1_q8_i8_4x16,%function
/* x0 A panels, x1 B panels, x2 output, x3 output row stride,
 * w4 positive count of 32-channel groups. */
d1_q8_i8_4x16:
    stp d8,d9,[sp,#-32]!
    stp d10,d11,[sp,#16]
    movi v0.4s,#0
    mov x8,x2
    mov w7,#4
.Lq8_i8_init:
    stp q0,q0,[x8]
    stp q0,q0,[x8,#32]
    add x8,x8,x3
    subs w7,w7,#1
    b.ne .Lq8_i8_init
.Lq8_i8_group:
    .irp acc,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
        movi v\acc\().4s,#0
    .endr
    add x5,x0,#8
    add x6,x1,#32
    mov w7,#8
.Lq8_i8_pairs:
    q8_i8_pair
    q8_i8_pair
    subs w7,w7,#1
    b.ne .Lq8_i8_pairs
    mov x8,x2
    q8_i8_finish 0,16,17,18,19
    q8_i8_finish 1,20,21,22,23
    q8_i8_finish 2,24,25,26,27
    q8_i8_finish 3,28,29,30,31
    add x0,x0,#136
    add x1,x1,#544
    subs w4,w4,#1
    b.ne .Lq8_i8_group
    ldp d10,d11,[sp,#16]
    ldp d8,d9,[sp],#32
    ret
.size d1_q8_i8_4x16,.-d1_q8_i8_4x16

/* Four rows, two adjacent signed-byte channels per lane. */
.macro q8_i8_pack_half offset
    ldur q0,[x8,#\offset]
    ldur q1,[x9,#\offset]
    ldur q2,[x10,#\offset]
    ldur q3,[x11,#\offset]
    zip1 v4.8h,v0.8h,v1.8h
    zip2 v5.8h,v0.8h,v1.8h
    zip1 v6.8h,v2.8h,v3.8h
    zip2 v7.8h,v2.8h,v3.8h
    zip1 v0.4s,v4.4s,v6.4s
    zip2 v1.4s,v4.4s,v6.4s
    zip1 v2.4s,v5.4s,v7.4s
    zip2 v3.4s,v5.4s,v7.4s
    stp q0,q1,[x1],#32
    stp q2,q3,[x1],#32
.endm

.global d1_pack_q8_i8_a4
.type d1_pack_q8_i8_a4,%function
/* x0 raw Q8_0 activations, x1 packed output, x2 raw row byte stride,
 * w3 positive number of 32-channel blocks, w4 positive number of row tiles. */
d1_pack_q8_i8_a4:
.Lq8_pack_rows:
    mov x8,x0
    add x9,x8,x2
    add x10,x9,x2
    add x11,x10,x2
    mov w13,w3
.Lq8_pack_group:
    ldrh w14,[x8]
    ldrh w15,[x9]
    ldrh w16,[x10]
    ldrh w17,[x11]
    strh w14,[x1]
    strh w15,[x1,#2]
    strh w16,[x1,#4]
    strh w17,[x1,#6]
    add x1,x1,#8
    q8_i8_pack_half 2
    q8_i8_pack_half 18
    add x8,x8,#34
    add x9,x9,#34
    add x10,x10,#34
    add x11,x11,#34
    subs w13,w13,#1
    b.ne .Lq8_pack_group
    add x0,x0,x2,lsl #2
    subs w4,w4,#1
    b.ne .Lq8_pack_rows
    ret
.size d1_pack_q8_i8_a4,.-d1_pack_q8_i8_a4

.section .note.GNU-stack,"",%progbits
