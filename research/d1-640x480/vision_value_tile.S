/* Ordered FP32 value accumulation for the 1200/300-key vision shapes. */
.arch armv8-a
.include "src/common.inc"
.text

.balign 32
.global d1_pack_value_head
.type d1_pack_value_head,%function
/* x0 contiguous [key][64] values, x1 packed [8][key][8], w2 key count. */
d1_pack_value_head:
    mov w3,#8
.Lvalue_head_panel:
    mov x4,x0
    mov w5,w2
.Lvalue_head_key:
    ldp q0,q1,[x4]
    stp q0,q1,[x1],#32
    add x4,x4,#256
    subs w5,w5,#1
    b.ne .Lvalue_head_key
    add x0,x0,#32
    subs w3,w3,#1
    b.ne .Lvalue_head_panel
    ret
.size d1_pack_value_head,.-d1_pack_value_head

.global d1_pack_value_probabilities
.type d1_pack_value_probabilities,%function
/* x0 probabilities, x1 [key][12], w2 positive key count divisible by 4,
 * x3 probability row stride in bytes. */
d1_pack_value_probabilities:
    mov w10,#3
.Lvalue_prob_group:
    mov x4,x0
    add x5,x4,x3
    add x6,x5,x3
    add x7,x6,x3
    mov x8,x1
    mov w9,w2
.Lvalue_prob_keys:
    ldr q0,[x4],#16
    ldr q1,[x5],#16
    ldr q2,[x6],#16
    ldr q3,[x7],#16
    trn1 v4.4s,v0.4s,v1.4s
    trn2 v5.4s,v0.4s,v1.4s
    trn1 v6.4s,v2.4s,v3.4s
    trn2 v7.4s,v2.4s,v3.4s
    trn1 v0.2d,v4.2d,v6.2d
    trn1 v1.2d,v5.2d,v7.2d
    trn2 v2.2d,v4.2d,v6.2d
    trn2 v3.2d,v5.2d,v7.2d
    str q0,[x8]
    str q1,[x8,#48]
    str q2,[x8,#96]
    str q3,[x8,#144]
    add x8,x8,#192
    subs w9,w9,#4
    b.ne .Lvalue_prob_keys
    add x0,x0,x3,lsl #2
    add x1,x1,#16
    subs w10,w10,#1
    b.ne .Lvalue_prob_group
    ret
.size d1_pack_value_probabilities,.-d1_pack_value_probabilities

.balign 32
.global d1_vision_value12
.type d1_vision_value12,%function
/* x0 twelve rows of probabilities, x1 packed value head, x2 output,
 * w3 key count divisible by 4, x4 output row stride in bytes,
 * x5 private scratch (12*K + 12*64 floats), x6 probability row stride. */
d1_vision_value12:
    ENTER
    mov x19,x1
    mov x20,x2
    mov w21,w3
    mov x22,x4
    mov x23,x5
    mov w24,#48
    madd x24,x21,x24,x23
    mov x1,x23
    mov w2,w21
    mov x3,x6
    bl d1_pack_value_probabilities
    mov w25,#0
.Lvalue12_k:
    sub w26,w21,w25
    mov w0,#120
    cmp w26,w0
    csel w26,w26,w0,lo
    mov w27,#0
.Lvalue12_n:
    mov w0,#48
    madd x0,x25,x0,x23
    madd w1,w27,w21,w25
    add x1,x19,x1,lsl #5
    add x2,x24,x27,lsl #5
    mov w3,#256
    mov w4,w26
    cmp w25,#0
    cset w5,ne
    mov w6,#1
    bl d1_fmla_12x8
    add w27,w27,#1
    cmp w27,#8
    b.ne .Lvalue12_n
    add w25,w25,w26
    cmp w25,w21
    b.lo .Lvalue12_k

    /* Compact accumulator rows avoid 4 KiB-stride set conflicts. */
    mov w0,#12
.Lvalue12_write:
    mov x1,x20
    .rept 8
    ldp q0,q1,[x24],#32
    stp q0,q1,[x1],#32
    .endr
    add x20,x20,x22
    subs w0,w0,#1
    b.ne .Lvalue12_write
    LEAVE
.size d1_vision_value12,.-d1_vision_value12
.section .note.GNU-stack,"",%progbits
