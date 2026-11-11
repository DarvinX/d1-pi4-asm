/* d1 consumes only the logits of the labels in its decision question. */
.arch armv8-a+crc
.include "src/common.inc"
.text
.balign 64
.global d1_label_logits
.type d1_label_logits,%function
/* x0 hidden[2048], x1 Q6_K row base, x2 label IDs, x3 logits,
 * w4 number of labels, x5 aligned 2560-byte query scratch.
 * The caller validates IDs against the bound vocabulary. */
d1_label_logits:
    cbz w4,.Lempty
    ENTER
    mov x19,x1
    mov x20,x2
    mov x21,x3
    mov w22,w4
    mov x23,x5
    mov x1,x23
    mov w2,#2048
    bl lfm_q8_pack_native
    mov x24,#1680 /* 2048 / 256 * sizeof(block_q6_K) */
.Llabel:
    ldr w0,[x20],#4
    madd x0,x0,x24,x19
    mov x1,x23
    mov w2,#2048
    bl lfm_q6_q8_dot_h19
    str s0,[x21],#4
    subs w22,w22,#1
    b.ne .Llabel
    LEAVE
.Lempty:
    ret
.size d1_label_logits,.-d1_label_logits
.section .note.GNU-stack,"",%progbits
