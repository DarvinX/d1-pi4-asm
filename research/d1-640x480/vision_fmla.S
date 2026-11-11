/* Cache-blocked vision projection primitives for Cortex-A72.
 * A 12 x 8 tile holds 24 independent four-lane accumulators.
 * The packer preserves the original Q8_0 weights and activation scales. */

.text
.arch armv8-a

.macro d1_fmla_dequant_vector reg
    scvtf v\reg\().4s, v\reg\().4s
    fmul v\reg\().4s, v\reg\().4s, v0.4s
    str q\reg, [x1]
    add x1, x1, x3
.endm

.macro d1_fmla_decode_four source
    sxtl v21.8h, v\source\().8b
    sxtl2 v22.8h, v\source\().16b
    sxtl v23.4s, v21.4h
    sxtl2 v24.4s, v21.8h
    sxtl v25.4s, v22.4h
    sxtl2 v26.4s, v22.8h
    d1_fmla_dequant_vector 23
    d1_fmla_dequant_vector 24
    d1_fmla_dequant_vector 25
    d1_fmla_dequant_vector 26
.endm

.macro d1_fmla_store_halfwords source
    sxtl v21.8h,v\source\().8b
    sxtl2 v22.8h,v\source\().16b
    stp q21,q22,[x3],#32
.endm

.macro d1_fmla_interleave_four halfwords=0
    zip1 v5.16b, v1.16b, v2.16b
    zip2 v6.16b, v1.16b, v2.16b
    zip1 v7.16b, v3.16b, v4.16b
    zip2 v16.16b, v3.16b, v4.16b
    zip1 v17.8h, v5.8h, v7.8h
    zip2 v18.8h, v5.8h, v7.8h
    zip1 v19.8h, v6.8h, v16.8h
    zip2 v20.8h, v6.8h, v16.8h
    .if \halfwords
        d1_fmla_store_halfwords 17
        d1_fmla_store_halfwords 18
        d1_fmla_store_halfwords 19
        d1_fmla_store_halfwords 20
    .else
        d1_fmla_decode_four 17
        d1_fmla_decode_four 18
        d1_fmla_decode_four 19
        d1_fmla_decode_four 20
    .endif
.endm

.macro d1_fmla_read_bytes halfwords=0
    ldr q1, [x0], #16
    ldr q2, [x5], #16
    ldr q3, [x6], #16
    ldr q4, [x7], #16
    d1_fmla_interleave_four \halfwords
.endm

.balign 32
.global d1_pack_fmla_q8_4
.type d1_pack_fmla_q8_4, %function
/* x0: first of four Q8_0 rows, x1: output, x2: input row stride (bytes),
 * x3: output channel stride (bytes), w4: channels (positive multiple of 32).
 * Writes four adjacent float lanes for each channel. */
d1_pack_fmla_q8_4:
    add x5, x0, x2
    add x6, x5, x2
    add x7, x6, x2
.Lpack_block:
    ldr h0, [x0], #2
    ldr h1, [x5], #2
    ldr h2, [x6], #2
    ldr h3, [x7], #2
    fcvt s0, h0
    fcvt s1, h1
    fcvt s2, h2
    fcvt s3, h3
    mov v0.s[1], v1.s[0]
    mov v0.s[2], v2.s[0]
    mov v0.s[3], v3.s[0]
    .rept 2
        d1_fmla_read_bytes
    .endr
    subs w4, w4, #32
    b.ne .Lpack_block
    ret
.size d1_pack_fmla_q8_4, .-d1_pack_fmla_q8_4

.balign 32
.global d1_pack_fmla_q8k_4
.type d1_pack_fmla_q8k_4, %function
/* Same interface as the Q8_0 packer, for d1's 320-byte Q8_K block. */
d1_pack_fmla_q8k_4:
    add x5,x0,x2
    add x6,x5,x2
    add x7,x6,x2
.Lpack_q8k_block:
    ldr s0,[x0],#32
    ldr s1,[x5],#32
    ldr s2,[x6],#32
    ldr s3,[x7],#32
    mov v0.s[1],v1.s[0]
    mov v0.s[2],v2.s[0]
    mov v0.s[3],v3.s[0]
    mov w8,#16
.Lpack_q8k_bytes:
    d1_fmla_read_bytes
    subs w8,w8,#1
    b.ne .Lpack_q8k_bytes
    add x0,x0,#32
    add x5,x5,#32
    add x6,x6,#32
    add x7,x7,#32
    subs w4,w4,#256
    b.ne .Lpack_q8k_block
    ret
.size d1_pack_fmla_q8k_4, .-d1_pack_fmla_q8k_4

.balign 32
.global d1_pack_q8k_i16_4
.type d1_pack_q8k_i16_4,%function
/* Four Q8_K rows at x0 and stride x2. Write their FP32 base scales
 * to x1 and 256 groups of four signed halfword codes to x3. */
d1_pack_q8k_i16_4:
    add x5,x0,x2
    add x6,x5,x2
    add x7,x6,x2
    ldr s0,[x0],#32
    ldr s1,[x5],#32
    ldr s2,[x6],#32
    ldr s3,[x7],#32
    mov v0.s[1],v1.s[0]
    mov v0.s[2],v2.s[0]
    mov v0.s[3],v3.s[0]
    str q0,[x1]
    mov w8,#16
.Lpack_q8k_i16:
    d1_fmla_read_bytes 1
    subs w8,w8,#1
    b.ne .Lpack_q8k_i16
    ret
.size d1_pack_q8k_i16_4,.-d1_pack_q8k_i16_4

.macro d1_fmla_q6_codes dst,high,ptr,lo,hi,shift,nibble
    ldr q\dst,[\ptr,#\lo]
    ldr q\high,[\ptr,#\hi]
    .if \nibble
    ushr v\dst\().16b,v\dst\().16b,#4
    .else
    and v\dst\().16b,v\dst\().16b,v31.16b
    .endif
    .if \shift
    ushr v\high\().16b,v\high\().16b,#\shift
    .endif
    and v\high\().16b,v\high\().16b,v17.16b
    shl v\high\().16b,v\high\().16b,#4
    orr v\dst\().16b,v\dst\().16b,v\high\().16b
    sub v\dst\().16b,v\dst\().16b,v18.16b
.endm

.macro d1_fmla_q6_scale dst,ptr,offset,base
    ldrsb w8,[\ptr,#\offset]
    scvtf s\dst,w8
    fmul s\dst,s\dst,s\base
.endm

.macro d1_fmla_q6_group lo,hi,shift,nibble,scale
    movi v17.16b,#3
    movi v18.16b,#32
    d1_fmla_q6_codes 1,5,x0,\lo,\hi,\shift,\nibble
    d1_fmla_q6_codes 2,6,x5,\lo,\hi,\shift,\nibble
    d1_fmla_q6_codes 3,7,x6,\lo,\hi,\shift,\nibble
    d1_fmla_q6_codes 4,16,x7,\lo,\hi,\shift,\nibble
    d1_fmla_q6_scale 0,x0,\scale,27
    d1_fmla_q6_scale 5,x5,\scale,28
    d1_fmla_q6_scale 6,x6,\scale,29
    d1_fmla_q6_scale 7,x7,\scale,30
    mov v0.s[1],v5.s[0]
    mov v0.s[2],v6.s[0]
    mov v0.s[3],v7.s[0]
    d1_fmla_interleave_four
.endm

.macro d1_fmla_q6_half lo,hi,scale
    d1_fmla_q6_group \lo,\hi,0,0,\scale
    d1_fmla_q6_group (\lo+16),(\hi+16),0,0,(\scale+1)
    d1_fmla_q6_group (\lo+32),\hi,2,0,(\scale+2)
    d1_fmla_q6_group (\lo+48),(\hi+16),2,0,(\scale+3)
    d1_fmla_q6_group \lo,\hi,4,1,(\scale+4)
    d1_fmla_q6_group (\lo+16),(\hi+16),4,1,(\scale+5)
    d1_fmla_q6_group (\lo+32),\hi,6,1,(\scale+6)
    d1_fmla_q6_group (\lo+48),(\hi+16),6,1,(\scale+7)
.endm

.balign 32
.global d1_pack_fmla_q6k_4
.type d1_pack_fmla_q6k_4, %function
/* Original Q6_K rows, same interface as the Q8_K packer. */
d1_pack_fmla_q6k_4:
    add x5,x0,x2
    add x6,x5,x2
    add x7,x6,x2
    movi v31.16b,#15
.Lpack_q6k_block:
    ldr h27,[x0,#208]
    ldr h28,[x5,#208]
    ldr h29,[x6,#208]
    ldr h30,[x7,#208]
    fcvt s27,h27
    fcvt s28,h28
    fcvt s29,h29
    fcvt s30,h30
    d1_fmla_q6_half 0,128,192
    d1_fmla_q6_half 64,160,200
    add x0,x0,#210
    add x5,x5,#210
    add x6,x6,#210
    add x7,x7,#210
    subs w4,w4,#256
    b.ne .Lpack_q6k_block
    ret
.size d1_pack_fmla_q6k_4, .-d1_pack_fmla_q6k_4

.macro d1_fmla_rows a,r0,r1,r2,r3,r4,r5,r6,r7,integer=0
    .if \integer
    smlal v\r0\().4s,v0.4h,v\a\().h[0]
    smlal2 v\r1\().4s,v0.8h,v\a\().h[0]
    smlal v\r2\().4s,v0.4h,v\a\().h[1]
    smlal2 v\r3\().4s,v0.8h,v\a\().h[1]
    smlal v\r4\().4s,v0.4h,v\a\().h[2]
    smlal2 v\r5\().4s,v0.8h,v\a\().h[2]
    smlal v\r6\().4s,v0.4h,v\a\().h[3]
    smlal2 v\r7\().4s,v0.8h,v\a\().h[3]
    .else
    fmla v\r0\().4s, v0.4s, v\a\().s[0]
    fmla v\r1\().4s, v1.4s, v\a\().s[0]
    fmla v\r2\().4s, v0.4s, v\a\().s[1]
    fmla v\r3\().4s, v1.4s, v\a\().s[1]
    fmla v\r4\().4s, v0.4s, v\a\().s[2]
    fmla v\r5\().4s, v1.4s, v\a\().s[2]
    fmla v\r6\().4s, v0.4s, v\a\().s[3]
    fmla v\r7\().4s, v1.4s, v\a\().s[3]
    .endif
.endm

.macro d1_fmla_step compact=0
    .if \compact
    ldr q1,[x7],#16
    saddw v0.4s,v7.4s,v1.4h
    saddw2 v1.4s,v7.4s,v1.8h
    .else
    ldp q0, q1, [x7], #32
    .endif
    ldp q2, q3, [x0], #32
    ldr q4, [x0], #16
    .if \compact
    fsub v0.4s,v0.4s,v7.4s
    fsub v1.4s,v1.4s,v7.4s
    .endif
    d1_fmla_rows 2, 8, 9, 10, 11, 12, 13, 14, 15
    d1_fmla_rows 3, 16, 17, 18, 19, 20, 21, 22, 23
    d1_fmla_rows 4, 24, 25, 26, 27, 28, 29, 30, 31
.endm

.macro d1_fmla_transfer operation,first,second,scale=0,accumulate=0
    .if \scale == 2
        ldr s4,[x10],#4
        fmul v2.4s,v5.4s,v4.s[0]
        fmul v3.4s,v6.4s,v4.s[0]
        scvtf v\first\().4s,v\first\().4s
        scvtf v\second\().4s,v\second\().4s
        .if \accumulate
        ldp q0,q1,[x8]
        .else
        movi v0.4s,#0
        movi v1.4s,#0
        .endif
        fmla v0.4s,v2.4s,v\first\().4s
        fmla v1.4s,v3.4s,v\second\().4s
        stp q0,q1,[x8]
    .elseif \scale == 1
        .if \accumulate
        ldp q0,q1,[x8]
        fmla v0.4s,v\first\().4s,v5.4s
        fmla v1.4s,v\second\().4s,v6.4s
        stp q0,q1,[x8]
        .else
        fmul v\first\().4s,v\first\().4s,v5.4s
        fmul v\second\().4s,v\second\().4s,v6.4s
        stp q\first,q\second,[x8]
        .endif
    .else
    \operation q\first, q\second, [x8]
    .endif
    add x8, x8, x3
.endm

.macro d1_fmla_transfer_tile operation,scale=0,accumulate=0
    mov x8, x2
    d1_fmla_transfer \operation,8,9,\scale,\accumulate
    d1_fmla_transfer \operation,10,11,\scale,\accumulate
    d1_fmla_transfer \operation,12,13,\scale,\accumulate
    d1_fmla_transfer \operation,14,15,\scale,\accumulate
    d1_fmla_transfer \operation,16,17,\scale,\accumulate
    d1_fmla_transfer \operation,18,19,\scale,\accumulate
    d1_fmla_transfer \operation,20,21,\scale,\accumulate
    d1_fmla_transfer \operation,22,23,\scale,\accumulate
    d1_fmla_transfer \operation,24,25,\scale,\accumulate
    d1_fmla_transfer \operation,26,27,\scale,\accumulate
    d1_fmla_transfer \operation,28,29,\scale,\accumulate
    d1_fmla_transfer \operation,30,31,\scale,\accumulate
.endm

.balign 32
.global d1_fmla_12x8
.type d1_fmla_12x8, %function
/* x0: consecutive 12-position panels, x1: one 8-weight panel,
 * x2: destination, x3: output row stride (bytes), w4: positive even K,
 * w5: accumulate existing outputs if nonzero, w6: positive panel count. */
d1_fmla_12x8:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
.Lfmla_panel:
    cbz w5, .Lfmla_zero
    d1_fmla_transfer_tile ldp
    b .Lfmla_inputs
.Lfmla_zero:
    .irp reg,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
        movi v\reg\().4s, #0
    .endr
.Lfmla_inputs:
    mov x7, x1
    mov w9, w4
.Lfmla_k:
    d1_fmla_step
    d1_fmla_step
    subs w9, w9, #2
    b.ne .Lfmla_k
    d1_fmla_transfer_tile stp
    mov x2, x8
    subs w6, w6, #1
    b.ne .Lfmla_panel
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8, d9, [sp], #64
    ret
.size d1_fmla_12x8, .-d1_fmla_12x8

.ifdef D1_STRASSEN_FANOUT
.macro d1_fmla_add_tile_row first,second
    ldp q0,q1,[x8]
    eor v2.16b,v\first\().16b,v4.16b
    eor v3.16b,v\second\().16b,v4.16b
    fadd v0.4s,v0.4s,v2.4s
    fadd v1.4s,v1.4s,v3.4s
    stp q0,q1,[x8]
    add x8,x8,x3
.endm

.balign 32
.global d1_fmla_12x8_fanout
.type d1_fmla_12x8_fanout,%function
/* x0 packed activations, x1 packed weights, x2 array of 16-byte output
 * descriptors (address, sign bit), x3 output stride, w4 even K, w5 count. */
d1_fmla_12x8_fanout:
    stp d8,d9,[sp,#-64]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    stp d14,d15,[sp,#48]
    .irp reg,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    movi v\reg\().4s,#0
    .endr
    mov x7,x1
.Lfanout_k:
    d1_fmla_step
    d1_fmla_step
    subs w4,w4,#2
    b.ne .Lfanout_k
.Lfanout_destination:
    ldp x8,x9,[x2],#16
    dup v4.4s,w9
    d1_fmla_add_tile_row 8,9
    d1_fmla_add_tile_row 10,11
    d1_fmla_add_tile_row 12,13
    d1_fmla_add_tile_row 14,15
    d1_fmla_add_tile_row 16,17
    d1_fmla_add_tile_row 18,19
    d1_fmla_add_tile_row 20,21
    d1_fmla_add_tile_row 22,23
    d1_fmla_add_tile_row 24,25
    d1_fmla_add_tile_row 26,27
    d1_fmla_add_tile_row 28,29
    d1_fmla_add_tile_row 30,31
    subs w5,w5,#1
    b.ne .Lfanout_destination
    ldp d10,d11,[sp,#16]
    ldp d12,d13,[sp,#32]
    ldp d14,d15,[sp,#48]
    ldp d8,d9,[sp],#64
    ret
.size d1_fmla_12x8_fanout,.-d1_fmla_12x8_fanout
.endif

.balign 32
.global d1_fmla_i16_12x8
.type d1_fmla_i16_12x8,%function
/* Same arguments as d1_fmla_12x8. The weight panel starts with eight
 * FP16 base scales, followed by K rows of signed Q6 scale × code. */
d1_fmla_i16_12x8:
    stp d8,d9,[sp,#-64]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    stp d14,d15,[sp,#48]
    ldr q5,[x1],#16
    fcvtl2 v6.4s,v5.8h
    fcvtl v5.4s,v5.4h
    /* At this bias, adding the signed integer to the float encoding
     * keeps the exponent fixed. Subtraction recovers the exact integer. */
    mov w10,#0x1000
    movk w10,#0x4b00,lsl #16
    dup v7.4s,w10
.Li16_panel:
    .irp reg,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
        movi v\reg\().4s,#0
    .endr
    mov x7,x1
    mov w9,w4
.Li16_k:
    d1_fmla_step 1
    d1_fmla_step 1
    subs w9,w9,#2
    b.ne .Li16_k
    cbz w5,.Li16_first
    d1_fmla_transfer_tile stp,1,1
    b .Li16_done_panel
.Li16_first:
    d1_fmla_transfer_tile stp,1,0
.Li16_done_panel:
    mov x2,x8
    subs w6,w6,#1
    b.ne .Li16_panel
    ldp d10,d11,[sp,#16]
    ldp d12,d13,[sp,#32]
    ldp d14,d15,[sp,#48]
    ldp d8,d9,[sp],#64
    ret
.size d1_fmla_i16_12x8,.-d1_fmla_i16_12x8

.macro d1_q6_i16_exact_step
    ldr q0,[x7],#16
    ldr d1,[x0],#8
    ldr d2,[x11],#8
    ldr d3,[x12],#8
    d1_fmla_rows 1,8,9,10,11,12,13,14,15,1
    d1_fmla_rows 2,16,17,18,19,20,21,22,23,1
    d1_fmla_rows 3,24,25,26,27,28,29,30,31,1
.endm

.balign 32
.global d1_q6_i16_exact_12x8
.type d1_q6_i16_exact_12x8,%function
/* Same result and weight arguments as the compact FMLA tile.
 * Each activation panel has 12 FP32 scales and [3][K][4] halfwords.
 * K is positive, even and at most 256. */
d1_q6_i16_exact_12x8:
    stp d8,d9,[sp,#-64]!
    stp d10,d11,[sp,#16]
    stp d12,d13,[sp,#32]
    stp d14,d15,[sp,#48]
    ldr q5,[x1],#16
    fcvtl2 v6.4s,v5.8h
    fcvtl v5.4s,v5.4h
.Lq6_exact_panel:
    .irp reg,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
        movi v\reg\().4s,#0
    .endr
    mov x10,x0
    add x0,x0,#48
    add x11,x0,x4,lsl #3
    add x12,x11,x4,lsl #3
    mov x7,x1
    mov w9,w4
.Lq6_exact_k:
    d1_q6_i16_exact_step
    d1_q6_i16_exact_step
    subs w9,w9,#2
    b.ne .Lq6_exact_k
    cbz w5,.Lq6_exact_first
    d1_fmla_transfer_tile stp,2,1
    b .Lq6_exact_done_panel
.Lq6_exact_first:
    d1_fmla_transfer_tile stp,2,0
.Lq6_exact_done_panel:
    mov x0,x12
    mov x2,x8
    subs w6,w6,#1
    b.ne .Lq6_exact_panel
    ldp d10,d11,[sp,#16]
    ldp d12,d13,[sp,#32]
    ldp d14,d15,[sp,#48]
    ldp d8,d9,[sp],#64
    ret
.size d1_q6_i16_exact_12x8,.-d1_q6_i16_exact_12x8

.section .note.GNU-stack,"",%progbits
