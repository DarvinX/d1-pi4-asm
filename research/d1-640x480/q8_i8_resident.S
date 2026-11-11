/* Use the D178 layout. Keep eight output accumulators per position
 * in registers, processing the two halves of the 16-output tile in turn. */
.arch armv8-a
.text

/* Weight codes are checked at pack time to lie in [-127,127].
 * Even for signed-byte activations, two products fit in each halfword. */
.macro q8_res_mac lane,lo,hi
    dup v4.8h,v2.h[\lane]
    dup v5.8h,v3.h[\lane]
    smull v8.8h,v0.8b,v4.8b
    smull2 v9.8h,v0.16b,v4.16b
    smlal v8.8h,v1.8b,v5.8b
    smlal2 v9.8h,v1.16b,v5.16b
    sadalp v\lo\().4s,v8.8h
    sadalp v\hi\().4s,v9.8h
.endm

.macro q8_res_finish lane,dot0,dot1,out0,out1
    ldr h4,[x0,#2*\lane]
    fcvt s4,h4
    fmul v5.4s,v0.4s,v4.s[0]
    fmul v6.4s,v1.4s,v4.s[0]
    scvtf v\dot0\().4s,v\dot0\().4s
    scvtf v\dot1\().4s,v\dot1\().4s
    fmla v\out0\().4s,v5.4s,v\dot0\().4s
    fmla v\out1\().4s,v6.4s,v\dot1\().4s
.endm

.balign 64
.global d1_q8_i8_resident_4x16
.type d1_q8_i8_resident_4x16,%function
/* Same layout and arguments as d1_q8_i8_4x16.
 * Weights must be checked to contain no -128 codes. */
d1_q8_i8_resident_4x16:
    stp d8,d9,[sp,#-16]!
    mov x12,x0
    mov x13,x1
    mov x14,x2
    mov w15,w4
    mov w16,#2
.Lq8_res_half:
    mov x0,x12
    mov x1,x13
    mov w4,w15
    .irp reg,24,25,26,27,28,29,30,31
    movi v\reg\().4s,#0
    .endr
.Lq8_res_group:
    .irp reg,16,17,18,19,20,21,22,23
    movi v\reg\().4s,#0
    .endr
    add x5,x0,#8
    add x6,x1,#32
    mov w7,#8
.Lq8_res_pairs:
    ldp d2,d3,[x5],#16
    ldr q0,[x6],#32
    ldr q1,[x6],#32
    q8_res_mac 0,16,17
    q8_res_mac 1,18,19
    q8_res_mac 2,20,21
    q8_res_mac 3,22,23
    subs w7,w7,#1
    b.ne .Lq8_res_pairs
    ldp d0,d1,[x1]
    fcvtl v0.4s,v0.4h
    fcvtl v1.4s,v1.4h
    q8_res_finish 0,16,17,24,25
    q8_res_finish 1,18,19,26,27
    q8_res_finish 2,20,21,28,29
    q8_res_finish 3,22,23,30,31
    add x0,x0,#136
    add x1,x1,#544
    subs w4,w4,#1
    b.ne .Lq8_res_group
    mov x8,x14
    stp q24,q25,[x8]
    add x8,x8,x3
    stp q26,q27,[x8]
    add x8,x8,x3
    stp q28,q29,[x8]
    add x8,x8,x3
    stp q30,q31,[x8]
    add x13,x13,#16
    add x14,x14,#32
    subs w16,w16,#1
    b.ne .Lq8_res_half
    ldp d8,d9,[sp],#16
    ret
.size d1_q8_i8_resident_4x16,.-d1_q8_i8_resident_4x16

.ifdef D1_Q8_LONG_PAIRS
/* Each four-channel step adds two products to each halfword lane.
 * With two steps and |w| <= 63 the bound is 32256.
 * With four steps and |w| <= 31 the bound is 31744.
 * With eight steps and |w| <= 15 the bound is 30720. */
.macro q8_long_mac lane,lo,hi,first
    dup v4.8h,v2.h[\lane]
    dup v5.8h,v3.h[\lane]
    .if \first
    smull v\lo\().8h,v0.8b,v4.8b
    smull2 v\hi\().8h,v0.16b,v4.16b
    .else
    smlal v\lo\().8h,v0.8b,v4.8b
    smlal2 v\hi\().8h,v0.16b,v4.16b
    .endif
    smlal v\lo\().8h,v1.8b,v5.8b
    smlal2 v\hi\().8h,v1.16b,v5.16b
.endm

.macro q8_long_step first=0,half_major=0
    ldp d2,d3,[x5],#16
    .if \half_major
    ldp q0,q1,[x6],#32
    .else
    ldr q0,[x6],#32
    ldr q1,[x6],#32
    .endif
    q8_long_mac 0,8,9,\first
    q8_long_mac 1,10,11,\first
    q8_long_mac 2,12,13,\first
    q8_long_mac 3,14,15,\first
.endm

.macro q8_long_widen op
    \op v16.4s,v8.8h
    \op v17.4s,v9.8h
    \op v18.4s,v10.8h
    \op v19.4s,v11.8h
    \op v20.4s,v12.8h
    \op v21.4s,v13.8h
    \op v22.4s,v14.8h
    \op v23.4s,v15.8h
.endm

.macro q8_long_four first
    q8_long_step \first,1
    q8_long_step 0,1
    q8_long_step 0,1
    q8_long_step 0,1
.endm

.macro D1_Q8_LONG_TILE name,half_major=0,certified=0
.balign 64
.global \name
.type \name,%function
/* Same ABI as the resident tile. Call only on validated narrow panels. */
\name:
    stp d8,d9,[sp,#-64]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    stp d14,d15,[sp,#48]
    mov x12,x0
    mov x13,x1
    mov x14,x2
    mov w15,w4
    mov w16,#2
.Lq8_long_half\@:
    mov x0,x12
    mov x1,x13
    mov w4,w15
    movi v24.4s,#0
    movi v25.4s,#0
    movi v26.4s,#0
    movi v27.4s,#0
    movi v28.4s,#0
    movi v29.4s,#0
    movi v30.4s,#0
    movi v31.4s,#0
.Lq8_long_group\@:
    .if \certified
    /* The first FP16 scale's sign bit certifies that every lane's
     * absolute weight sum is at most 255. Clear it before scaling. */
    add x5,x0,#8
    add x6,x1,#16
    ldrh w9,[x1]
    tbnz w9,#15,.Lq8_certified\@
    q8_long_four 1
    q8_long_widen saddlp
    q8_long_four 1
    q8_long_widen sadalp
    b .Lq8_long_scales\@
.Lq8_certified\@:
    q8_long_four 1
    q8_long_four 0
    q8_long_widen saddlp
.Lq8_long_scales\@:
    .else
    movi v16.4s,#0
    movi v17.4s,#0
    movi v18.4s,#0
    movi v19.4s,#0
    movi v20.4s,#0
    movi v21.4s,#0
    movi v22.4s,#0
    movi v23.4s,#0
    add x5,x0,#8
    .if \half_major
    add x6,x1,#16
    .else
    add x6,x1,#32
    .endif
    mov w7,#(8/D1_Q8_LONG_PAIRS)
.Lq8_long_pairs\@:
    q8_long_step 1,\half_major
    q8_long_step 0,\half_major
    .if D1_Q8_LONG_PAIRS >= 4
    q8_long_step 0,\half_major
    q8_long_step 0,\half_major
    .endif
    .if D1_Q8_LONG_PAIRS == 8
    q8_long_step 0,\half_major
    q8_long_step 0,\half_major
    q8_long_step 0,\half_major
    q8_long_step 0,\half_major
    .endif
    q8_long_widen sadalp
    subs w7,w7,#1
    b.ne .Lq8_long_pairs\@
    .endif
    ldp d0,d1,[x1]
    .if \certified
    bic v0.4h,#0x80,lsl #8
    .endif
    fcvtl v0.4s,v0.4h
    fcvtl v1.4s,v1.4h
    q8_res_finish 0,16,17,24,25
    q8_res_finish 1,18,19,26,27
    q8_res_finish 2,20,21,28,29
    q8_res_finish 3,22,23,30,31
    add x0,x0,#136
    .if \half_major
    add x1,x1,#272
    .else
    add x1,x1,#544
    .endif
    subs w4,w4,#1
    b.ne .Lq8_long_group\@
    mov x8,x14
    stp q24,q25,[x8]
    add x8,x8,x3
    stp q26,q27,[x8]
    add x8,x8,x3
    stp q28,q29,[x8]
    add x8,x8,x3
    stp q30,q31,[x8]
    .if \half_major
    mov x13,x1
    .else
    add x13,x13,#16
    .endif
    add x14,x14,#32
    subs w16,w16,#1
    b.ne .Lq8_long_half\@
    ldp d10,d11,[sp,#16]
    ldp d12,d13,[sp,#32]
    ldp d14,d15,[sp,#48]
    ldp d8,d9,[sp],#64
    ret
.size \name,.-\name
.endm
D1_Q8_LONG_TILE d1_q8_i8_long_4x16
.ifdef D1_Q8_HALF_LAYOUT
D1_Q8_LONG_TILE d1_q8_i8_half_4x16,1
.ifdef D1_Q8_CERTIFIED_LAYOUT
D1_Q8_LONG_TILE d1_q8_i8_certified_4x16,1,1
.endif
.endif
.endif
.section .note.GNU-stack,"",%progbits
