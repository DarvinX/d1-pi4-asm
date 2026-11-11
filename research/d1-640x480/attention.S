/* One 128-channel d1 attention head. The layer scheduler supplies the key span. */
.arch armv8-a+crc
.include "src/common.inc"
.text
.balign 64
.global d1_attention_head
.type d1_attention_head,%function
/*
 * x0 query, x1 keys, x2 values, x3 output, x4 scratch,
 * w5 key count (>0), x6 key/value stride in bytes.
 * The scratch capacity is count rounded up to 4 floats.
 * For global attention include the whole sequence. For sliding-window
 * attention pass the causal window and offset the key/value pointers.
 */
d1_attention_head:
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,x4
    mov w24,w5
    mov x25,x6
    mov w26,#0
    ldr s8,.Lscale
    ldr w0,=0xff800000
    fmov s9,w0
.Lscores:
    mov x0,x19
    madd x1,x26,x25,x20
    mov w2,#128
    bl lfm_dot_native
    fmul s0,s0,s8
    fmax s9,s9,s0
    str s0,[x23,x26,lsl #2]
    add w26,w26,#1
    cmp w26,w24
    b.lo .Lscores

    mov x0,x23
    mov w1,w24
    fmov s0,s9
    bl lfm_softmax_native

    mov x27,#0
.Lvalues:
    mov x0,x23
    add x1,x21,x27
    mov w2,w24
    mov x3,x25
    bl lfm_value_native
    str q0,[x22,x27]
    add x27,x27,#16
    cmp x27,#512
    b.lo .Lvalues
    LEAVE
.size d1_attention_head,.-d1_attention_head
.balign 4
.Lscale: .float 0.08838834764831845 /* 1/sqrt(128) */

#ifndef D1_LINK_EXISTING_NATIVE_MATH
.include "src/native-math.inc"
#endif
.section .note.GNU-stack,"",%progbits
