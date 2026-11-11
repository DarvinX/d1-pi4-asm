/* Model-specific RoPE and 2×2 merger for 640×480 still images. */
.arch armv8-a
.include "src/common.inc"
.text

.global d1_vision_rope_rows
.type d1_vision_rope_rows,%function
d1_vision_rope_rows: /* Q or K, 1200 × (cos32,sin32), heads per position */
    mov w3,#1200
.Lrope_pos:
    mov w4,w2
.Lrope_head:
    mov w5,#8
    mov x6,x1
    mov x7,x0
.Lrope_lane:
    ldr q0,[x7]
    ldr q1,[x7,#128]
    ldr q2,[x6]
    ldr q3,[x6,#128]
    fmul v4.4s,v0.4s,v2.4s
    fmul v5.4s,v0.4s,v3.4s
    fmls v4.4s,v1.4s,v3.4s
    fmla v5.4s,v1.4s,v2.4s
    str q5,[x7,#128]
    str q4,[x7],#16
    add x6,x6,#16
    subs w5,w5,#1
    b.ne .Lrope_lane
    add x0,x0,#256
    subs w4,w4,#1
    b.ne .Lrope_head
    add x1,x1,#256
    subs w3,w3,#1
    b.ne .Lrope_pos
    ret
.size d1_vision_rope_rows,.-d1_vision_rope_rows

.macro VISION_SUM acc,values
    fcvtl v6.2d,v\values\().2s
    fcvtl2 v7.2d,v\values\().4s
    fadd v\acc\().2d,v\acc\().2d,v6.2d
    fadd v\acc\().2d,v\acc\().2d,v7.2d
.endm

.type d1_vision_layernorm,%function
d1_vision_layernorm: /* one 1024-wide row in place, weights, bias */
    mov x3,x0
    mov w4,#256
    movi v16.2d,#0
.Lnorm_mean:
    ldr q0,[x3],#16
    VISION_SUM 16,0
    subs w4,w4,#1
    b.ne .Lnorm_mean
    faddp d16,v16.2d
    fcvt s16,d16
    ldr s18,.Linv1024
    fmul s16,s16,s18
    dup v16.4s,v16.s[0]
    mov x3,x0
    mov w4,#256
    movi v17.2d,#0
.Lnorm_var:
    ldr q0,[x3]
    fsub v0.4s,v0.4s,v16.4s
    str q0,[x3],#16
    fmul v0.4s,v0.4s,v0.4s
    VISION_SUM 17,0
    subs w4,w4,#1
    b.ne .Lnorm_var
    faddp d17,v17.2d
    fcvt s17,d17
    fmul s17,s17,s18
    ldr s18,.Lnorm_eps
    fadd s17,s17,s18
    fsqrt s17,s17
    fmov s18,#1.0
    fdiv s17,s18,s17
    dup v17.4s,v17.s[0]
    mov w4,#256
.Lnorm_out:
    ldr q0,[x0]
    ldr q1,[x1],#16
    ldr q2,[x2],#16
    fmul v0.4s,v0.4s,v17.4s
    fmul v0.4s,v0.4s,v1.4s
    fadd v0.4s,v0.4s,v2.4s
    str q0,[x0],#16
    subs w4,w4,#1
    b.ne .Lnorm_out
    ret
.size d1_vision_layernorm,.-d1_vision_layernorm

.global d1_vision_norm_merge
.type d1_vision_norm_merge,%function
d1_vision_norm_merge: /* encoder rows (mutated), gamma, beta, 300 merged rows */
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov w23,#1200
.Lnorm_all:
    mov x1,x20
    mov x2,x21
    mov x24,x0
    bl d1_vision_layernorm
    add x0,x24,#1,lsl #12
    subs w23,w23,#1
    b.ne .Lnorm_all
    mov w23,#15
.Lmerge_y:
    mov w24,#20
.Lmerge_x:
    mov x0,x22
    mov x1,x19
    mov x2,#8192
    bl memcpy
    add x0,x22,#2,lsl #12
    add x1,x19,#40,lsl #12
    mov x2,#8192
    bl memcpy
    add x22,x22,#4,lsl #12
    add x19,x19,#2,lsl #12
    subs w24,w24,#1
    b.ne .Lmerge_x
    add x19,x19,#40,lsl #12
    subs w23,w23,#1
    b.ne .Lmerge_y
    LEAVE
.size d1_vision_norm_merge,.-d1_vision_norm_merge

.global d1_vision_gelu_erf
.type d1_vision_gelu_erf,%function
d1_vision_gelu_erf: /* x0 input/output, w1 count */
    ENTER
    mov x19,x0
    mov w20,w1
    ldr s9,.Lsqrt_inv
.Lgelu:
    ldr s8,[x19]
    fmul s0,s8,s9
    bl erff
    fmov s1,#1.0
    fmov s2,#0.5
    fadd s0,s0,s1
    fmul s8,s8,s2
    fmul s0,s0,s8
    str s0,[x19],#4
    subs w20,w20,#1
    b.ne .Lgelu
    LEAVE
.size d1_vision_gelu_erf,.-d1_vision_gelu_erf
.balign 4
.Linv1024: .float 0.0009765625
.Lnorm_eps: .float 0.000001
.Lsqrt_inv: .float 0.7071067811865475
.section .note.GNU-stack,"",%progbits
