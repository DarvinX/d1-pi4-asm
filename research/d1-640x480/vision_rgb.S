/* Exact 640×480 LFM3-VL input staging. Caller supplies decoded RGB bytes. */
.arch armv8-a+crc
.text
.balign 64

/* Keep the reference's rounding: float(byte)/255, subtract 0.5, divide by 0.5. */
.macro NORMALIZE vec
    ucvtf v\vec\().4s,v\vec\().4s
    fdiv v\vec\().4s,v\vec\().4s,v30.4s
    fsub v\vec\().4s,v\vec\().4s,v31.4s
    fadd v\vec\().4s,v\vec\().4s,v\vec\().4s
.endm

.macro WRITE_CHANNEL input, output
    ushll v3.8h,v\input\().8b,#0
    ushll2 v4.8h,v\input\().16b,#0
    ushll v16.4s,v3.4h,#0
    ushll2 v17.4s,v3.8h,#0
    ushll v18.4s,v4.4h,#0
    ushll2 v19.4s,v4.8h,#0
    NORMALIZE 16
    NORMALIZE 17
    NORMALIZE 18
    NORMALIZE 19
    st1 {v16.4s-v19.4s},[\output],#64
.endm

.global d1_rgb640x480
.type d1_rgb640x480,%function
/* x0: interleaved RGB8, x1: 3 contiguous 640×480 FP32 planes. */
d1_rgb640x480:
    mov x2,#0xc000
    movk x2,#0x12,lsl #16 /* one plane: 640 * 480 * sizeof(float) */
    add x3,x1,x2
    add x4,x3,x2
    mov w5,#19200 /* 16 pixels per iteration */
    ldr s30,.Lbyte_max
    dup v30.4s,v30.s[0]
    fmov v31.4s,#0.5
.Lrgb:
    ld3 {v0.16b-v2.16b},[x0],#48
    WRITE_CHANNEL 0,x1
    WRITE_CHANNEL 1,x3
    WRITE_CHANNEL 2,x4
    subs w5,w5,#1
    b.ne .Lrgb
    ret
.size d1_rgb640x480,.-d1_rgb640x480

.balign 64
.global d1_patch640x480
.type d1_patch640x480,%function
/* x0: interleaved RGB8, x1: 1200 contiguous [3][16][16] FP16 patch rows.
 * Both temporal patch convolutions consume this same still-image input. */
d1_patch640x480:
    sub sp,sp,#512
    mov w8,#0
    ldr s30,.Lbyte_max
    fmov s31,#0.5
.Llookup:
    ucvtf s0,w8
    fdiv s0,s0,s30
    fsub s0,s0,s31
    fadd s0,s0,s0
    fcvt h0,s0
    str h0,[sp,w8,uxtw #1]
    add w8,w8,#1
    cmp w8,#256
    b.ne .Llookup
    mov x2,#30720 /* 16 image rows * 640 * 3 */
    mov w3,#30
.Lpatch_band:
    mov w4,#40
    mov x5,x0
.Lpatch:
    mov x6,x5
    add x7,x1,#512
    add x8,x1,#1024
    mov w9,#16
.Lpatch_row:
    mov x10,x6
    mov w11,#16
.Lpatch_pixel:
    ldrb w12,[x10],#1
    ldrb w13,[x10],#1
    ldrb w14,[x10],#1
    ldrh w12,[sp,w12,uxtw #1]
    ldrh w13,[sp,w13,uxtw #1]
    ldrh w14,[sp,w14,uxtw #1]
    strh w12,[x1],#2
    strh w13,[x7],#2
    strh w14,[x8],#2
    subs w11,w11,#1
    b.ne .Lpatch_pixel
    add x6,x6,#1920
    subs w9,w9,#1
    b.ne .Lpatch_row
    mov x1,x8
    add x5,x5,#48
    subs w4,w4,#1
    b.ne .Lpatch
    add x0,x0,x2
    subs w3,w3,#1
    b.ne .Lpatch_band
    add sp,sp,#512
    ret
.size d1_patch640x480,.-d1_patch640x480

.balign 4
.Lbyte_max: .float 255.0
.section .note.GNU-stack,"",%progbits
