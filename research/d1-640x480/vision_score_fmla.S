/* Use the same FP32 FMLA panel as vision value12 for the dense score matrix. */
.arch armv8-a
.include "src/common.inc"
.text
/* x0 Q rows, x1 packed K, x2 score rows, x3 12x64 scratch,
 * x4 input row stride. Scores use a fixed 1200-float stride. */
.macro SCORE12 name,raw=0,pooled=0
.global \name
.type \name,%function
\name:
    ENTER
    mov x19,x1
    mov x20,x2
    mov x21,x3
    .if \pooled
    mov w25,w5
    mov x24,x6
    .endif
    mov x1,x3
    mov w2,#64
    mov x3,x4
    bl d1_pack_value_probabilities
    mov x22,x20
    .if \pooled
    lsr w23,w25,#3
    .else
    mov w23,#150
    .endif
.Lscore_panels\@:
    mov x0,x21
    mov x1,x19
    mov x2,x22
    mov w3,#4800
    mov w4,#64
    mov w5,#0
    mov w6,#1
    bl d1_fmla_12x8
    add x19,x19,#2048
    add x22,x22,#32
    subs w23,w23,#1
    b.ne .Lscore_panels\@
    mov w23,#12
.Lscore_softmax\@:
    mov x0,x20
    .if \pooled
    mov x2,x24
    fmov v16.4s,#0.125
    mov w1,#0xff800000
    dup v17.4s,w1
    mov v18.16b,v17.16b
    lsr w1,w25,#3
.Lscore_pooled_scale\@:
    ldp q0,q1,[x0]
    ldp q2,q3,[x2],#32
    fmul v0.4s,v0.4s,v16.4s
    fmul v1.4s,v1.4s,v16.4s
    fadd v0.4s,v0.4s,v2.4s
    fadd v1.4s,v1.4s,v3.4s
    fmax v17.4s,v17.4s,v0.4s
    fmax v18.4s,v18.4s,v1.4s
    stp q0,q1,[x0],#32
    subs w1,w1,#1
    b.ne .Lscore_pooled_scale\@
    fmax v17.4s,v17.4s,v18.4s
    fmaxv s0,v17.4s
    mov x0,x20
    mov w1,w25
    .else
    .if !\raw
    fmov v16.4s,#0.125
    .endif
    mov w1,#0xff800000
    dup v17.4s,w1
    mov v18.16b,v17.16b
    mov v19.16b,v17.16b
    mov v20.16b,v17.16b
    mov w1,#75
.Lscore_scale\@:
    ldp q0,q1,[x0]
    ldp q2,q3,[x0,#32]
    .if !\raw
    fmul v0.4s,v0.4s,v16.4s
    fmul v1.4s,v1.4s,v16.4s
    fmul v2.4s,v2.4s,v16.4s
    fmul v3.4s,v3.4s,v16.4s
    .endif
    fmax v17.4s,v17.4s,v0.4s
    fmax v18.4s,v18.4s,v1.4s
    fmax v19.4s,v19.4s,v2.4s
    fmax v20.4s,v20.4s,v3.4s
    .if !\raw
    stp q0,q1,[x0]
    stp q2,q3,[x0,#32]
    .endif
    add x0,x0,#64
    subs w1,w1,#1
    b.ne .Lscore_scale\@
    fmax v17.4s,v17.4s,v18.4s
    fmax v19.4s,v19.4s,v20.4s
    fmax v17.4s,v17.4s,v19.4s
    fmaxv s0,v17.4s
    mov x0,x20
    mov w1,#1200
    .endif
    .if \raw
    bl d1_softmax_raw_score12
    .else
    bl lfm_softmax_native
    .endif
    add x20,x20,#4096
    add x20,x20,#704
    subs w23,w23,#1
    b.ne .Lscore_softmax\@
    LEAVE
.size \name,.-\name
.endm

SCORE12 d1_attention_score12
.ifdef D1_VISION_POOLED12
/* w5 is the padded key count, x6 holds one log multiplicity per key. */
SCORE12 d1_attention_score12_pooled,0,1
.endif
.ifdef D1_SCORE12_RAW_SOFTMAX
SCORE12 d1_attention_score12_raw,1
.endif
.balign 32
.global d1_pack_score_keys8
.type d1_pack_score_keys8,%function
/* x0 contiguous [1200][64] keys, x1 packed [150][64][8]. */
d1_pack_score_keys8:
    mov w10,#150
    .ifdef D1_VISION_POOLED12
    b .Lscore_key_panel
.global d1_pack_score_keys8_counted
.type d1_pack_score_keys8_counted,%function
d1_pack_score_keys8_counted: /* w2 positive padded key count */
    lsr w10,w2,#3
    .endif
.Lscore_key_panel:
    mov x11,x0
    mov x12,x1
    mov w13,#2
.Lscore_key_half:
    mov x3,x11
    add x4,x3,#256
    add x5,x4,#256
    add x6,x5,#256
    mov x7,x12
    mov w8,#16
.Lscore_key_channels:
    ldr q0,[x3],#16
    ldr q1,[x4],#16
    ldr q2,[x5],#16
    ldr q3,[x6],#16
    trn1 v4.4s,v0.4s,v1.4s
    trn2 v5.4s,v0.4s,v1.4s
    trn1 v6.4s,v2.4s,v3.4s
    trn2 v7.4s,v2.4s,v3.4s
    trn1 v0.2d,v4.2d,v6.2d
    trn1 v1.2d,v5.2d,v7.2d
    trn2 v2.2d,v4.2d,v6.2d
    trn2 v3.2d,v5.2d,v7.2d
    str q0,[x7]
    str q1,[x7,#32]
    str q2,[x7,#64]
    str q3,[x7,#96]
    add x7,x7,#128
    subs w8,w8,#1
    b.ne .Lscore_key_channels
    add x11,x11,#1024
    add x12,x12,#16
    subs w13,w13,#1
    b.ne .Lscore_key_half
    add x0,x0,#2048
    add x1,x1,#2048
    subs w10,w10,#1
    b.ne .Lscore_key_panel
    ret
.size d1_pack_score_keys8,.-d1_pack_score_keys8
.ifdef D1_VISION_POOLED12
.size d1_pack_score_keys8_counted,.-d1_pack_score_keys8_counted
.endif
.section .note.GNU-stack,"",%progbits
