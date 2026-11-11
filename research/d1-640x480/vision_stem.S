/* Still-image patch stem. The offline asset combines the two temporal
 * filters. Every invocation recomputes patches from the supplied RGB. */
.arch armv8-a
.include "src/common.inc"
.text

.global d1_stem_prepare
.type d1_stem_prepare,%function
d1_stem_prepare: /* x0: RGB, x1: FP32 patches, x2: FP16 scratch */
    stp x19,x20,[sp,#-32]!
    str x30,[sp,#16]
    mov x19,x1
    mov x20,x2
    mov x1,x2
    bl d1_patch640x480
    ldr w2,=115200 /* 1200 * 768 / 8 */
.Lconvert:
    ldr q0,[x20],#16
    fcvtl v1.4s,v0.4h
    fcvtl2 v2.4s,v0.8h
    stp q1,q2,[x19],#32
    subs w2,w2,#1
    b.ne .Lconvert
    ldr x30,[sp,#16]
    ldp x19,x20,[sp],#32
    ret
.size d1_stem_prepare,.-d1_stem_prepare

.macro STEM_FMA a,b,c,d
    ldr q0,[x0],#16
    ldr q1,[x1],#16
    ldr q2,[x5],#16
    ldr q3,[x6],#16
    ldr q4,[x7],#16
    fmla v\a\().4s,v0.4s,v1.4s
    fmla v\b\().4s,v0.4s,v2.4s
    fmla v\c\().4s,v0.4s,v3.4s
    fmla v\d\().4s,v0.4s,v4.4s
.endm
.macro STEM_SUM a,b,c,d,native
    .if \native
    fadd v\a\().4s,v\a\().4s,v\c\().4s
    fadd v\b\().4s,v\b\().4s,v\d\().4s
    fadd v\a\().4s,v\a\().4s,v\b\().4s
    .else
    fadd v\a\().4s,v\a\().4s,v\b\().4s
    fadd v\c\().4s,v\c\().4s,v\d\().4s
    fadd v\a\().4s,v\a\().4s,v\c\().4s
    .endif
    faddp v\a\().4s,v\a\().4s,v\a\().4s
    faddp s\a,v\a\().2s
.endm
.macro STEM_FMA16
    STEM_FMA 16,20,24,28
    STEM_FMA 17,21,25,29
    STEM_FMA 18,22,26,30
    STEM_FMA 19,23,27,31
.endm
.macro STEM_TILE name,native
.balign 64
.global \name
.type \name,%function
\name:
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
    .if \native
    .rept 4
    STEM_FMA16
    .endr
    .else
1:  STEM_FMA16
    subs w4,w4,#16
    b.ne 1b
    .endif
    STEM_SUM 16,17,18,19,\native
    STEM_SUM 20,21,22,23,\native
    STEM_SUM 24,25,26,27,\native
    STEM_SUM 28,29,30,31,\native
    stp s16,s20,[x3]
    stp s24,s28,[x3,#8]
    ret
.size \name,.-\name
.endm

/* weight row, four input rows, byte stride, out[4]; x4 is K for the stem. */
STEM_TILE d1_dot4_f32,0
/* Attention has K=64 and uses the exact lfm_dot_native reduction. */
STEM_TILE d1_dot4_f32_native64,1

.global d1_stem_project_slice
.type d1_stem_project_slice,%function
d1_stem_project_slice: /* weights, patches, output, first channel, channel count */
    cbz w4,.Lslice_empty
    ENTER
    mov x5,#3072
    madd x19,x3,x5,x0
    mov x20,x1
    add x21,x2,x3,lsl #2
    mov w22,w4
    mov w23,#300
    sub sp,sp,#16
.Lslice_patches:
    mov x24,x19
    mov x25,x21
    mov w26,w22
.Lslice_rows:
    mov x0,x24
    mov x1,x20
    mov x2,#3072
    mov x3,sp
    mov w4,#768
    bl d1_dot4_f32
    ldp s0,s1,[sp]
    ldp s2,s3,[sp,#8]
    str s0,[x25]
    str s1,[x25,#4096]
    str s2,[x25,#8192]
    str s3,[x25,#12288]
    add x24,x24,#3072
    add x25,x25,#4
    subs w26,w26,#1
    b.ne .Lslice_rows
    add x20,x20,#3,lsl #12
    add x21,x21,#4,lsl #12
    subs w23,w23,#1
    b.ne .Lslice_patches
    add sp,sp,#16
    LEAVE
.Lslice_empty:
    ret
.size d1_stem_project_slice,.-d1_stem_project_slice

.global d1_vision_rms_rows
.type d1_vision_rms_rows,%function
d1_vision_rms_rows: /* output, input, weights, width, rows */
    cbz w4,.Lrms_empty
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov w22,w3
    mov w23,w4
.Lrms_rows:
    mov x0,x19
    mov x1,x20
    mov x2,x21
    mov w3,w22
    ldr s0,.Lvision_eps
    bl lfm_rms_wide
    add x19,x19,x22,lsl #2
    add x20,x20,x22,lsl #2
    subs w23,w23,#1
    b.ne .Lrms_rows
    LEAVE
.Lrms_empty:
    ret
.size d1_vision_rms_rows,.-d1_vision_rms_rows
.balign 4
.Lvision_eps: .float 0.00001
.section .note.GNU-stack,"",%progbits
