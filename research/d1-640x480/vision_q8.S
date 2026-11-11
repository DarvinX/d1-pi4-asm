/* Q8_0 for the LFM3-VL projector. One 34-byte block covers 32 values. */
.arch armv8-a+crc
.text
.balign 64
.global d1_pack_q8_0
.global d1_pack_q8_0_narrow
.type d1_pack_q8_0,%function
.type d1_pack_q8_0_narrow,%function

.macro QUANTIZE lane
    fmul v\lane\().4s,v\lane\().4s,v25.4s
    fcvtns v\lane\().4s,v\lane\().4s
.endm

/* x0: input FP32, x1: Q8_0 output, w2: elements, multiple of 32. */
d1_pack_q8_0:
    ldr s30,.Lqmax
    b .Lpack_start
/* Same layout with a seven-bit signed payload. */
d1_pack_q8_0_narrow:
    ldr s30,.Lqmax_narrow
.Lpack_start:
    cbz w2,.Lpack_done
    fmov s29,#1.0
.Lpack:
    ld1 {v0.4s-v3.4s},[x0],#64
    ld1 {v4.4s-v7.4s},[x0],#64
    fabs v16.4s,v0.4s
    fabs v17.4s,v1.4s
    fabs v18.4s,v2.4s
    fabs v19.4s,v3.4s
    fabs v20.4s,v4.4s
    fabs v21.4s,v5.4s
    fabs v22.4s,v6.4s
    fabs v23.4s,v7.4s
    fmax v16.4s,v16.4s,v17.4s
    fmax v18.4s,v18.4s,v19.4s
    fmax v20.4s,v20.4s,v21.4s
    fmax v22.4s,v22.4s,v23.4s
    fmax v16.4s,v16.4s,v18.4s
    fmax v20.4s,v20.4s,v22.4s
    fmax v16.4s,v16.4s,v20.4s
    fmaxv s16,v16.4s
    fdiv s16,s16,s30
    fcvt h17,s16
    str h17,[x1],#2
    fmov s25,wzr
    fcmp s16,#0.0
    b.eq .Lscale_ready
    fdiv s25,s29,s16
.Lscale_ready:
    dup v25.4s,v25.s[0]
    QUANTIZE 0
    QUANTIZE 1
    QUANTIZE 2
    QUANTIZE 3
    QUANTIZE 4
    QUANTIZE 5
    QUANTIZE 6
    QUANTIZE 7
    sqxtn v16.4h,v0.4s
    sqxtn2 v16.8h,v1.4s
    sqxtn v17.4h,v2.4s
    sqxtn2 v17.8h,v3.4s
    sqxtn v18.4h,v4.4s
    sqxtn2 v18.8h,v5.4s
    sqxtn v19.4h,v6.4s
    sqxtn2 v19.8h,v7.4s
    sqxtn v20.8b,v16.8h
    sqxtn2 v20.16b,v17.8h
    sqxtn v21.8b,v18.8h
    sqxtn2 v21.16b,v19.8h
    st1 {v20.16b,v21.16b},[x1],#32
    subs w2,w2,#32
    b.ne .Lpack
.Lpack_done:
    ret
.size d1_pack_q8_0,.-d1_pack_q8_0
.size d1_pack_q8_0_narrow,.-d1_pack_q8_0_narrow
.Lqmax: .float 127.0
.Lqmax_narrow: .float 63.0

.macro LOAD_WEIGHT
    ldr h23,[x0],#2
    fcvt s23,h23
    ld1 {v0.16b,v1.16b},[x0],#32
.endm

.macro DOT_ACTIVATION row,accum
    ldr h24,[\row],#2
    fcvt s24,h24
    fmul s24,s23,s24
    ld1 {v2.16b,v3.16b},[\row],#32
    smull v4.8h,v0.8b,v2.8b
    smull2 v5.8h,v0.16b,v2.16b
    smull v6.8h,v1.8b,v3.8b
    smull2 v7.8h,v1.16b,v3.16b
    saddlp v16.4s,v4.8h
    sadalp v16.4s,v5.8h
    sadalp v16.4s,v6.8h
    sadalp v16.4s,v7.8h
    scvtf v16.4s,v16.4s
    fmla v\accum\().4s,v16.4s,v24.s[0]
.endm

/* With quantized activations in [-127,127], two signed byte products
 * sum to at most 32512 in magnitude even for a -128 weight. Widen
 * after combining the low and high halves. The FP32 lane sums and
 * their accumulation order are unchanged. */
.macro DOT_ACTIVATION_PAIR row,accum
    ldr h24,[\row],#2
    fcvt s24,h24
    fmul s24,s23,s24
    ld1 {v2.16b,v3.16b},[\row],#32
    smull v4.8h,v0.8b,v2.8b
    smull v6.8h,v1.8b,v3.8b
    smlal2 v4.8h,v0.16b,v2.16b
    smlal2 v6.8h,v1.16b,v3.16b
    saddlp v16.4s,v4.8h
    sadalp v16.4s,v6.8h
    scvtf v16.4s,v16.4s
    fmla v\accum\().4s,v16.4s,v24.s[0]
.endm

.macro LOAD_ACTIVATION row,scale,lo,hi
    ldr h\scale,[\row],#2
    fcvt s\scale,h\scale
    fmul s\scale,s23,s\scale
    ld1 {v\lo\().16b,v\hi\().16b},[\row],#32
.endm

.macro FOLD_POSITION product0,product1,work,scale,accum
    saddlp v\work\().4s,v\product0\().8h
    sadalp v\work\().4s,v\product1\().8h
    scvtf v\work\().4s,v\work\().4s
    fmla v\accum\().4s,v\work\().4s,v\scale\().s[0]
.endm

.macro FOLD_NARROW product,work,scale,accum
    saddlp v\work\().4s,v\product\().8h
    scvtf v\work\().4s,v\work\().4s
    fmla v\accum\().4s,v\work\().4s,v\scale\().s[0]
.endm

.macro DOT_BLOCK_ILP a,b,c,d,narrow=0
    LOAD_ACTIVATION x1,18,2,3
    LOAD_ACTIVATION x5,19,4,5
    LOAD_ACTIVATION x6,22,6,7
    LOAD_ACTIVATION x7,24,8,9
    smull v10.8h,v0.8b,v2.8b
    smull v12.8h,v0.8b,v4.8b
    smull v14.8h,v0.8b,v6.8b
    smull v16.8h,v0.8b,v8.8b
    .if \narrow
    /* Four products fit in a signed halfword: 4 * 128 * 63 = 32256. */
    smlal2 v10.8h,v0.16b,v2.16b
    smlal2 v12.8h,v0.16b,v4.16b
    smlal2 v14.8h,v0.16b,v6.16b
    smlal2 v16.8h,v0.16b,v8.16b
    smlal v10.8h,v1.8b,v3.8b
    smlal v12.8h,v1.8b,v5.8b
    smlal v14.8h,v1.8b,v7.8b
    smlal v16.8h,v1.8b,v9.8b
    smlal2 v10.8h,v1.16b,v3.16b
    smlal2 v12.8h,v1.16b,v5.16b
    smlal2 v14.8h,v1.16b,v7.16b
    smlal2 v16.8h,v1.16b,v9.16b
    FOLD_NARROW 10,2,18,\a
    FOLD_NARROW 12,4,19,\b
    FOLD_NARROW 14,6,22,\c
    FOLD_NARROW 16,8,24,\d
    .else
    smull v11.8h,v1.8b,v3.8b
    smull v13.8h,v1.8b,v5.8b
    smull v15.8h,v1.8b,v7.8b
    smull v17.8h,v1.8b,v9.8b
    smlal2 v10.8h,v0.16b,v2.16b
    smlal2 v12.8h,v0.16b,v4.16b
    smlal2 v14.8h,v0.16b,v6.16b
    smlal2 v16.8h,v0.16b,v8.16b
    smlal2 v11.8h,v1.16b,v3.16b
    smlal2 v13.8h,v1.16b,v5.16b
    smlal2 v15.8h,v1.16b,v7.16b
    smlal2 v17.8h,v1.16b,v9.16b
    FOLD_POSITION 10,11,2,18,\a
    FOLD_POSITION 12,13,4,19,\b
    FOLD_POSITION 14,15,6,22,\c
    FOLD_POSITION 16,17,8,24,\d
    .endif
.endm

.macro REDUCE_DOT even,odd
    faddp v\even\().4s,v\even\().4s,v\even\().4s
    faddp v\odd\().4s,v\odd\().4s,v\odd\().4s
    faddp s\even,v\even\().2s
    faddp s\odd,v\odd\().2s
    fadd s0,s\even,s\odd
.endm

.balign 64
.global d1_dot_q8_0
.type d1_dot_q8_0,%function
/* x0 and x1: Q8_0 rows, w2: positive element count, multiple of 64. */
d1_dot_q8_0:
    movi v20.4s,#0
    movi v21.4s,#0
.Ldot:
    LOAD_WEIGHT
    DOT_ACTIVATION x1,20
    LOAD_WEIGHT
    DOT_ACTIVATION x1,21
    subs w2,w2,#64
    b.ne .Ldot
    REDUCE_DOT 20,21
    ret
.size d1_dot_q8_0,.-d1_dot_q8_0

/* x0: one weight row, x1: four packed activation rows, x2: byte stride,
 * x3: four FP32 results, w4: positive element count, multiple of 64. */
.macro D1_DOT4_Q8 name,activation,ilp=0,narrow=0
.balign 64
.global \name
.type \name,%function
\name:
    .if \ilp
    stp d8,d9,[sp,#-64]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    stp d14,d15,[sp,#48]
    .endif
    add x5,x1,x2
    add x6,x5,x2
    add x7,x6,x2
    movi v20.4s,#0
    movi v21.4s,#0
    movi v25.4s,#0
    movi v26.4s,#0
    movi v27.4s,#0
    movi v28.4s,#0
    movi v29.4s,#0
    movi v30.4s,#0
1:
    LOAD_WEIGHT
    .if \ilp
    DOT_BLOCK_ILP 20,25,27,29,\narrow
    LOAD_WEIGHT
    DOT_BLOCK_ILP 21,26,28,30,\narrow
    .else
    \activation x1,20
    \activation x5,25
    \activation x6,27
    \activation x7,29
    LOAD_WEIGHT
    \activation x1,21
    \activation x5,26
    \activation x6,28
    \activation x7,30
    .endif
    subs w4,w4,#64
    b.ne 1b
    REDUCE_DOT 20,21
    str s0,[x3]
    REDUCE_DOT 25,26
    str s0,[x3,#4]
    REDUCE_DOT 27,28
    str s0,[x3,#8]
    REDUCE_DOT 29,30
    str s0,[x3,#12]
    .if \ilp
    ldp d14,d15,[sp,#48]
    ldp d12,d13,[sp,#32]
    ldp d10,d11,[sp,#16]
    ldp d8,d9,[sp],#64
    .endif
    ret
.size \name,.-\name
.endm

D1_DOT4_Q8 d1_dot4_q8_0,DOT_ACTIVATION
D1_DOT4_Q8 d1_dot4_q8_0_pair,DOT_ACTIVATION_PAIR
D1_DOT4_Q8 d1_dot4_q8_0_ilp,DOT_ACTIVATION_PAIR,1
D1_DOT4_Q8 d1_dot4_q8_0_narrow,DOT_ACTIVATION_PAIR,1,1

.macro LOAD_TWO_WEIGHTS
    ldr h10,[x0],#2
    ldr h11,[x8],#2
    fcvt s10,h10
    fcvt s11,h11
    ld1 {v0.16b,v1.16b},[x0],#32
    ld1 {v2.16b,v3.16b},[x8],#32
.endm

.macro DOT_ACTIVATION_TWO_WEIGHTS row,a,b
    ldr h12,[\row],#2
    fcvt s12,h12
    fmul s13,s10,s12
    fmul s14,s11,s12
    ld1 {v4.16b,v5.16b},[\row],#32
    smull v6.8h,v0.8b,v4.8b
    smull v7.8h,v1.8b,v5.8b
    smull v8.8h,v2.8b,v4.8b
    smull v9.8h,v3.8b,v5.8b
    smlal2 v6.8h,v0.16b,v4.16b
    smlal2 v7.8h,v1.16b,v5.16b
    smlal2 v8.8h,v2.16b,v4.16b
    smlal2 v9.8h,v3.16b,v5.16b
    FOLD_POSITION 6,7,15,13,\a
    FOLD_POSITION 8,9,15,14,\b
.endm

.balign 64
.global d1_dot4_q8_0_two_weights
.type d1_dot4_q8_0_two_weights,%function
/* x0 first weight row, x1 first activation, x2 activation stride,
 * x3 float output[2][4], w4 width, x5 weight row stride. */
d1_dot4_q8_0_two_weights:
    stp d8,d9,[sp,#-64]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    stp d14,d15,[sp,#48]
    add x8,x0,x5
    add x5,x1,x2
    add x6,x5,x2
    add x7,x6,x2
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
.Ltwo_weights:
    LOAD_TWO_WEIGHTS
    DOT_ACTIVATION_TWO_WEIGHTS x1,16,24
    DOT_ACTIVATION_TWO_WEIGHTS x5,18,26
    DOT_ACTIVATION_TWO_WEIGHTS x6,20,28
    DOT_ACTIVATION_TWO_WEIGHTS x7,22,30
    LOAD_TWO_WEIGHTS
    DOT_ACTIVATION_TWO_WEIGHTS x1,17,25
    DOT_ACTIVATION_TWO_WEIGHTS x5,19,27
    DOT_ACTIVATION_TWO_WEIGHTS x6,21,29
    DOT_ACTIVATION_TWO_WEIGHTS x7,23,31
    subs w4,w4,#64
    b.ne .Ltwo_weights
    REDUCE_DOT 16,17
    str s0,[x3]
    REDUCE_DOT 18,19
    str s0,[x3,#4]
    REDUCE_DOT 20,21
    str s0,[x3,#8]
    REDUCE_DOT 22,23
    str s0,[x3,#12]
    REDUCE_DOT 24,25
    str s0,[x3,#16]
    REDUCE_DOT 26,27
    str s0,[x3,#20]
    REDUCE_DOT 28,29
    str s0,[x3,#24]
    REDUCE_DOT 30,31
    str s0,[x3,#28]
    ldp d14,d15,[sp,#48]
    ldp d12,d13,[sp,#32]
    ldp d10,d11,[sp,#16]
    ldp d8,d9,[sp],#64
    ret
.size d1_dot4_q8_0_two_weights,.-d1_dot4_q8_0_two_weights

/* Experimental 256-channel scale layout. Rows contain all signed bytes
 * followed by one FP16 scale per group. The caller pads the row stride
 * for alignment and retains the pinned Q8_0 weights as the source. */
.balign 64
.global d1_pack_q8_256
.type d1_pack_q8_256,%function
d1_pack_q8_256: /* FP32 input, output, width (multiple of 256) */
    cbz w2,.Lpack256_done
    add x3,x1,w2,uxtw
    fmov s29,#1.0
    ldr s30,.Lpack256_max
.Lpack256_group:
    mov x4,x0
    mov w5,#16
    movi v16.4s,#0
    movi v17.4s,#0
    movi v18.4s,#0
    movi v19.4s,#0
.Lpack256_peak:
    ld1 {v0.4s-v3.4s},[x4],#64
    fabs v0.4s,v0.4s
    fabs v1.4s,v1.4s
    fabs v2.4s,v2.4s
    fabs v3.4s,v3.4s
    fmax v16.4s,v16.4s,v0.4s
    fmax v17.4s,v17.4s,v1.4s
    fmax v18.4s,v18.4s,v2.4s
    fmax v19.4s,v19.4s,v3.4s
    subs w5,w5,#1
    b.ne .Lpack256_peak
    fmax v16.4s,v16.4s,v17.4s
    fmax v18.4s,v18.4s,v19.4s
    fmax v16.4s,v16.4s,v18.4s
    fmaxv s16,v16.4s
    fdiv s16,s16,s30
    fcvt h17,s16
    str h17,[x3],#2
    fmov s25,wzr
    fcmp s16,#0.0
    b.eq .Lpack256_scale
    fdiv s25,s29,s16
.Lpack256_scale:
    dup v25.4s,v25.s[0]
    mov w5,#16
.Lpack256_bytes:
    ld1 {v0.4s-v3.4s},[x0],#64
    QUANTIZE 0
    QUANTIZE 1
    QUANTIZE 2
    QUANTIZE 3
    sqxtn v16.4h,v0.4s
    sqxtn2 v16.8h,v1.4s
    sqxtn v17.4h,v2.4s
    sqxtn2 v17.8h,v3.4s
    sqxtn v18.8b,v16.8h
    sqxtn2 v18.16b,v17.8h
    str q18,[x1],#16
    subs w5,w5,#1
    b.ne .Lpack256_bytes
    subs w2,w2,#256
    b.ne .Lpack256_group
.Lpack256_done:
    ret
.size d1_pack_q8_256,.-d1_pack_q8_256
.Lpack256_max: .float 127.0

.macro DOT256_FOLD sum,activation_scale,accum
    addv s\sum,v\sum\().4s
    scvtf s\sum,s\sum
    ldr h0,[\activation_scale],#2
    fcvt s0,h0
    fmul s0,s0,s24
    fmadd s\accum,s\sum,s0,s\accum
.endm

.balign 64
.global d1_dot4_q8_256
.type d1_dot4_q8_256,%function
d1_dot4_q8_256: /* weight, four packed activations, activation stride, out, width */
    stp d8,d9,[sp,#-64]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    stp d14,d15,[sp,#48]
    add x5,x1,x2
    add x6,x5,x2
    add x7,x6,x2
    add x8,x0,w4,uxtw
    add x9,x1,w4,uxtw
    add x10,x5,w4,uxtw
    add x11,x6,w4,uxtw
    add x12,x7,w4,uxtw
    fmov s25,wzr
    fmov s26,wzr
    fmov s27,wzr
    fmov s28,wzr
.Ldot256_group:
    movi v20.4s,#0
    movi v21.4s,#0
    movi v22.4s,#0
    movi v23.4s,#0
    mov w13,#8
.Ldot256_block:
    ld1 {v0.16b,v1.16b},[x0],#32
    ld1 {v2.16b,v3.16b},[x1],#32
    ld1 {v4.16b,v5.16b},[x5],#32
    ld1 {v6.16b,v7.16b},[x6],#32
    ld1 {v8.16b,v9.16b},[x7],#32
    smull v10.8h,v0.8b,v2.8b
    smull v12.8h,v0.8b,v4.8b
    smull v14.8h,v0.8b,v6.8b
    smull v16.8h,v0.8b,v8.8b
    smull v11.8h,v1.8b,v3.8b
    smull v13.8h,v1.8b,v5.8b
    smull v15.8h,v1.8b,v7.8b
    smull v17.8h,v1.8b,v9.8b
    smlal2 v10.8h,v0.16b,v2.16b
    smlal2 v12.8h,v0.16b,v4.16b
    smlal2 v14.8h,v0.16b,v6.16b
    smlal2 v16.8h,v0.16b,v8.16b
    smlal2 v11.8h,v1.16b,v3.16b
    smlal2 v13.8h,v1.16b,v5.16b
    smlal2 v15.8h,v1.16b,v7.16b
    smlal2 v17.8h,v1.16b,v9.16b
    sadalp v20.4s,v10.8h
    sadalp v21.4s,v12.8h
    sadalp v22.4s,v14.8h
    sadalp v23.4s,v16.8h
    sadalp v20.4s,v11.8h
    sadalp v21.4s,v13.8h
    sadalp v22.4s,v15.8h
    sadalp v23.4s,v17.8h
    subs w13,w13,#1
    b.ne .Ldot256_block
    ldr h24,[x8],#2
    fcvt s24,h24
    DOT256_FOLD 20,x9,25
    DOT256_FOLD 21,x10,26
    DOT256_FOLD 22,x11,27
    DOT256_FOLD 23,x12,28
    subs w4,w4,#256
    b.ne .Ldot256_group
    stp s25,s26,[x3]
    stp s27,s28,[x3,#8]
    ldp d14,d15,[sp,#48]
    ldp d12,d13,[sp,#32]
    ldp d10,d11,[sp,#16]
    ldp d8,d9,[sp],#64
    ret
.size d1_dot4_q8_256,.-d1_dot4_q8_256
.section .note.GNU-stack,"",%progbits
