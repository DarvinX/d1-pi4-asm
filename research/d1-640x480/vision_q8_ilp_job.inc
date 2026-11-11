/* Exact D184 weight panels, ordered as late FFNs, late operators, merger. */
.equ D1_Q8_ILP_TENSORS,38
.equ D1_Q8_ILP_PAYLOAD,106168320
.equ D1_Q8_ILP_FILE_BYTES,4096+D1_Q8_ILP_PAYLOAD

.text
.type d1_q8_ilp_map,%function
d1_q8_ilp_map:
    ENTER
    ADDR x0,d1_q8_ilp_env
    bl getenv
    cbz x0,.Lilp_map_done
    ldrb w1,[x0]
    cbz w1,.Lilp_map_done
    ldr x1,=D1_Q8_ILP_FILE_BYTES
    ADDR x2,d1_q8_ilp_magic
    ldr w3,=D1_VISION_CRC
    bl d1_panel_map_checked
    mov x19,x0
    add x20,x19,#4096
    mov x0,x20
    ldr x1,=D1_Q8_ILP_PAYLOAD
    mov w2,#-1
    bl d1_panel_crc
    mvn w0,w0
    ldr w1,[x19,#12]
    cmp w0,w1
    b.ne checksum_error
    bl getpagesize
    sub x2,x0,#1
    ADDR x0,d1_q8_ilp_page_mask
    str x2,[x0]
    mov x0,x20
    ldr x1,=D1_Q8_ILP_PAYLOAD
    bl d1_panel_discard
    cbnz w0,file_error
    ADDR x21,d1_vis_base
    ldr x21,[x21]
    ADDR x22,d1_vis_block_desc
    add x22,x22,#(8*13+2)*32
    ADDR x23,d1_q8_ilp_index
    mov w25,#6
.Lilp_map_ffns:
    mov x0,x22
    bl .Lilp_map_one
    add x0,x22,#32
    bl .Lilp_map_one
    add x0,x22,#64
    bl .Lilp_map_one
    add x22,x22,#13*32
    subs w25,w25,#1
    b.ne .Lilp_map_ffns
    ADDR x0,d1_q8_ilp_ffn_only_env
    bl getenv
    cbz x0,.Lilp_map_ops
    ldrb w0,[x0]
    cmp w0,#'1'
    b.eq .Lilp_map_count
.Lilp_map_ops:
    ADDR x22,d1_vis_block_desc
    add x22,x22,#(8*13+5)*32
    mov w25,#6
.Lilp_map_layer:
    mov x26,x22
    mov w24,#6
.Lilp_map_op_slot:
    ldr w0,[x26,#16]
    cmp w0,#8
    b.ne .Lilp_map_op_next
    mov x0,x26
    bl .Lilp_map_one
.Lilp_map_op_next:
    add x26,x26,#32
    subs w24,w24,#1
    b.ne .Lilp_map_op_slot
    add x22,x22,#13*32
    subs w25,w25,#1
    b.ne .Lilp_map_layer
    ADDR x0,d1_vis_up_desc
    bl .Lilp_map_one
    ADDR x0,d1_vis_down_desc
    bl .Lilp_map_one
    ldr x0,=D1_Q8_ILP_FILE_BYTES
    add x0,x19,x0
    cmp x0,x20
    b.ne checksum_error
.Lilp_map_count:
    ADDR x0,d1_q8_ilp_index
    sub x1,x23,x0
    lsr x1,x1,#4
    ADDR x0,d1_q8_ilp_count
    str w1,[x0]
.Lilp_map_done:
    LEAVE

.Lilp_map_one: /* x0 descriptor, x20 payload cursor, x21 projector, x23 index cursor */
    ldr x1,[x0]
    add x1,x21,x1
    stp x1,x20,[x23],#16
    ldp w1,w2,[x0,#8]
    mul w1,w1,w2
    lsr w1,w1,#5
    add x20,x20,x1,lsl #5
    add x20,x20,x1,lsl #2
    ret
.size d1_q8_ilp_map,.-d1_q8_ilp_map

.type d1_q8_ilp_prepare,%function
d1_q8_ilp_prepare: /* x0 matrix job with raw Q8_0 activations */
    ldr w1,[x0,#56]
    cmp w1,#D1_Q8_I8_POSITIONS
    b.ne .Lilp_unselected
    ADDR x1,d1_q8_ilp_count
    ldr w4,[x1]
    cbz w4,.Lilp_unselected
    ADDR x1,d1_vis_q8_ilp_enabled
    ldr w1,[x1]
    cbz w1,.Lilp_unselected
    ADDR x1,d1_q8_ilp_index
    ldr x3,[x0]
.Lilp_find:
    ldp x2,x5,[x1],#16
    cmp x2,x3
    b.eq .Lilp_selected
    subs w4,w4,#1
    b.ne .Lilp_find
.Lilp_unselected:
    mov w0,#0
    ret
.Lilp_selected:
    ENTER
    mov x19,x0
    ADDR x1,d1_q8_i8_for_job
    str x5,[x1]
    ldp w1,w2,[x19,#48]
    mul w1,w1,w2
    mov x0,x3
    ADDR x2,d1_q8_ilp_page_mask
    ldr x2,[x2]
    bl d1_panel_discard
    cbnz w0,file_error
    ldr x0,[x19,#8]
    ADDR x1,d1_fmla_activations
    ldr x2,[x19,#24]
    ldr w3,[x19,#40]
    lsr w3,w3,#5
    mov w4,#D1_Q8_I8_POSITIONS/4
    bl d1_q8_ilp_pack_a4
    ADDR x0,d1_q8_ilp_batches
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    ADDR x0,d1_fmla_this_job
    mov w1,#5
    str w1,[x0]
    mov w0,#1
    LEAVE
.size d1_q8_ilp_prepare,.-d1_q8_ilp_prepare

.type d1_q8_ilp_report,%function
d1_q8_ilp_report:
    ADDR x0,d1_q8_ilp_count
    ldr w0,[x0]
    cbz w0,.Lilp_report_done
    mov w0,#2
    ADDR x1,d1_q8_ilp_stats
    ADDR x2,d1_q8_ilp_batches
    ldr w2,[x2]
    mov w3,#D1_Q8_I8_POSITIONS
    mul w3,w2,w3
    b dprintf
.Lilp_report_done:
    ret
.size d1_q8_ilp_report,.-d1_q8_ilp_report
.ltorg

.section .rodata
.balign 8
d1_q8_ilp_magic: .ascii "D1Q8LP\001\000"
d1_q8_ilp_env: .asciz "D1_VISION_Q8_ILP_PANELS"
d1_q8_ilp_ffn_only_env: .asciz "D1_VISION_Q8_ILP_FFN_ONLY"
d1_q8_ilp_stats: .asciz "d1_vision_q8_ilp batches=%u rows=%u\n"
.bss
.balign 8
d1_q8_ilp_index: .skip D1_Q8_ILP_TENSORS*16
d1_q8_ilp_page_mask: .skip 8
d1_q8_ilp_count: .skip 4
d1_q8_ilp_batches: .skip 4
.text
