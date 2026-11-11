/* Fixed still-image 40 × 30 × 1024 3 × 3 gated depthwise convolution.
 * A zero halo removes all boundary tests from the channel loop.
 * Frozen taps are laid out as nine arrays of 1024 FP32 values.
 */
.arch armv8-a
.text

.macro VISION_DW_STAGE name,mapped=0,width=40,height=30
.global \name
.type \name,%function
\name: /* x0 projection, x1 halo [32][42][1024], x2 raster map if mapped */
    stp x19,x20,[sp,#-32]!
    str x30,[sp,#16]
    mov x19,x0
    mov x20,x1
    .if \mapped
    str x21,[sp,#24]
    mov x21,x2
    .endif
    mov x0,x1
    mov w1,#0
    .if \width == 40
    mov x2,#((\height+2)*(\width+2)*1024*4)
    .else
    mov w2,#((\height+2)*(\width+2))
    lsl x2,x2,#12
    .endif
    bl memset
    add x1,x20,#(\width+3),lsl #12
    mov x0,x19
    mov w2,#\height
.Lstage_y\@:
    mov w3,#\width
.Lstage_x\@:
    .if \mapped
    ldr w0,[x21],#4
    mov x11,#12288
    madd x0,x0,x11,x19
    .endif
    mov x5,x0
    add x6,x0,#2,lsl #12
    mov x7,x1
    mov w4,#256
.Lgate\@:
    ldr q0,[x5],#16
    ldr q1,[x6],#16
    fmul v0.4s,v0.4s,v1.4s
    str q0,[x7],#16
    subs w4,w4,#1
    b.ne .Lgate\@
    .if !\mapped
    add x0,x0,#3,lsl #12
    .endif
    add x1,x1,#1,lsl #12
    subs w3,w3,#1
    b.ne .Lstage_x\@
    add x1,x1,#2,lsl #12
    subs w2,w2,#1
    b.ne .Lstage_y\@
    .if \mapped
    ldr x21,[sp,#24]
    .endif
    ldr x30,[sp,#16]
    ldp x19,x20,[sp],#32
    ret
.size \name,.-\name
.endm

VISION_DW_STAGE d1_vision_dw_stage
VISION_DW_STAGE d1_vision_dw_stage_mapped,1

.macro D1_DW_TAP row,col,tap
    ldr q1,[\row,#(\col)*4096]
    ldr q2,[x11,#(\tap)*4096]
    fmla v0.4s,v1.4s,v2.4s
.endm

.macro VISION_DW_APPLY name,mapped=0
.balign 64
.global \name
.type \name,%function
\name: /* projection, halo, taps, full output, raster map if mapped */
    add x0,x0,#1,lsl #12 /* middle third */
    .if \mapped
    mov x14,x4
    .endif
    mov w4,#30
.Lapply_y\@:
    mov w5,#40
.Lapply_x\@:
    mov x8,x1
    add x9,x8,#42,lsl #12
    add x10,x9,#42,lsl #12
    mov x11,x2
    .if \mapped
    ldr w12,[x14],#4
    mov x13,#12288
    madd x12,x12,x13,x0
    .else
    mov x12,x0
    .endif
    mov w6,#256
.Lchannel\@:
    movi v0.4s,#0
    D1_DW_TAP x8,0,0
    D1_DW_TAP x8,1,1
    D1_DW_TAP x8,2,2
    D1_DW_TAP x9,0,3
    D1_DW_TAP x9,1,4
    D1_DW_TAP x9,2,5
    D1_DW_TAP x10,0,6
    D1_DW_TAP x10,1,7
    D1_DW_TAP x10,2,8
    ldr q1,[x12],#16
    fmul v0.4s,v0.4s,v1.4s
    str q0,[x3],#16
    add x8,x8,#16
    add x9,x9,#16
    add x10,x10,#16
    add x11,x11,#16
    subs w6,w6,#1
    b.ne .Lchannel\@
    .if !\mapped
    add x0,x0,#3,lsl #12
    .endif
    add x1,x1,#1,lsl #12
    subs w5,w5,#1
    b.ne .Lapply_x\@
    add x1,x1,#2,lsl #12
    subs w4,w4,#1
    b.ne .Lapply_y\@
    ret
.size \name,.-\name
.endm

VISION_DW_APPLY d1_vision_dw_apply
VISION_DW_APPLY d1_vision_dw_apply_mapped,1

.ifdef D1_VISION_COMPACT_DW
VISION_DW_STAGE d1_vision_dw_stage_quad,0,20,15

/* Compact halo lanes v16-v24 cover the union of the four stencils.
 * Each accumulator still visits the original nine taps in order. */
.macro D1_DW_QUAD_TAP tap,tl,tr,bl,br
    ldr q4,[x11,#(\tap)*4096]
    fmla v0.4s,v\tl\().4s,v4.4s
    fmla v1.4s,v\tr\().4s,v4.4s
    fmla v2.4s,v\bl\().4s,v4.4s
    fmla v3.4s,v\br\().4s,v4.4s
.endm

.balign 64
.global d1_vision_dw_apply_quad
.type d1_vision_dw_apply_quad,%function
/* x0 compact projection, x1 compact halo, x2 original taps, x3 compact result. */
d1_vision_dw_apply_quad:
    add x0,x0,#1,lsl #12
    fmov v25.4s,#0.25
    mov w4,#15
.Lquad_y:
    mov w5,#20
.Lquad_x:
    mov x8,x1
    add x9,x8,#22,lsl #12
    add x10,x9,#22,lsl #12
    mov x11,x2
    mov x12,x0
    mov w6,#256
.Lquad_channel:
    ldr q16,[x8]
    ldr q17,[x8,#4096]
    ldr q18,[x8,#8192]
    ldr q19,[x9]
    ldr q20,[x9,#4096]
    ldr q21,[x9,#8192]
    ldr q22,[x10]
    ldr q23,[x10,#4096]
    ldr q24,[x10,#8192]
    movi v0.4s,#0
    movi v1.4s,#0
    movi v2.4s,#0
    movi v3.4s,#0
    D1_DW_QUAD_TAP 0,16,17,19,20
    D1_DW_QUAD_TAP 1,17,17,20,20
    D1_DW_QUAD_TAP 2,17,18,20,21
    D1_DW_QUAD_TAP 3,19,20,19,20
    D1_DW_QUAD_TAP 4,20,20,20,20
    D1_DW_QUAD_TAP 5,20,21,20,21
    D1_DW_QUAD_TAP 6,19,20,22,23
    D1_DW_QUAD_TAP 7,20,20,23,23
    D1_DW_QUAD_TAP 8,20,21,23,24
    ldr q4,[x12],#16
    fmul v0.4s,v0.4s,v4.4s
    fmul v1.4s,v1.4s,v4.4s
    fmul v2.4s,v2.4s,v4.4s
    fmul v3.4s,v3.4s,v4.4s
    fadd v0.4s,v0.4s,v1.4s
    fadd v0.4s,v0.4s,v2.4s
    fadd v0.4s,v0.4s,v3.4s
    fmul v0.4s,v0.4s,v25.4s
    str q0,[x3],#16
    add x8,x8,#16
    add x9,x9,#16
    add x10,x10,#16
    add x11,x11,#16
    subs w6,w6,#1
    b.ne .Lquad_channel
    add x0,x0,#3,lsl #12
    add x1,x1,#1,lsl #12
    subs w5,w5,#1
    b.ne .Lquad_x
    add x1,x1,#2,lsl #12
    subs w4,w4,#1
    b.ne .Lquad_y
    ret
.size d1_vision_dw_apply_quad,.-d1_vision_dw_apply_quad
.endif
.section .note.GNU-stack,"",%progbits
