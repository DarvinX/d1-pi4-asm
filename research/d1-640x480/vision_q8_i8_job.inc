/* The 18 late grouped FFN weights have the same byte count as the Q8_0
 * originals. The runtime packs only the current decision's activations. */
.equ D1_Q8_I8_FFN_BYTES,2785280
.equ D1_Q8_I8_TENSORS,18
.equ D1_Q8_I8_FILE_BYTES,4096+D1_Q8_I8_TENSORS*D1_Q8_I8_FFN_BYTES
.equ D1_Q8_I8_POSITIONS,300
.equ D1_Q8_I8_A_BYTES,136
.equ D1_Q8_I8_W_BYTES,544

.text
.type d1_q8_i8_map,%function
d1_q8_i8_map:
    ENTER
    ADDR x0,d1_q8_i8_env
    bl getenv
    cbz x0,.Lq8_map_done
    ldrb w1,[x0]
    cbz w1,.Lq8_map_done
    ldr x1,=D1_Q8_I8_FILE_BYTES
    ADDR x2,d1_q8_i8_magic
    ldr w3,=D1_VISION_CRC
    bl d1_panel_map_checked
    mov x19,x0
    add x20,x19,#4096
    mov x0,x20
    ldr x1,=D1_Q8_I8_FILE_BYTES-4096
    mov w2,#-1
    bl d1_panel_crc
    mvn w0,w0
    ldr w1,[x19,#12]
    cmp w0,w1
    b.ne checksum_error
    mov w26,#(1<<D1_Q8_I8_TENSORS)-1
    ADDR x0,d1_q8_i8_mask_env
    bl getenv
    cbz x0,.Lq8_map_mask_ready
    mov x22,x0
    sub sp,sp,#16
    mov x1,sp
    mov w2,#0
    bl strtoul
    ldr x1,[sp]
    add sp,sp,#16
    cmp x1,x22
    b.eq .Lq8_map_mask_invalid
    ldrb w2,[x1]
    cbnz w2,.Lq8_map_mask_invalid
    cbz x0,.Lq8_map_mask_invalid
    cmp x0,x26
    b.hi .Lq8_map_mask_invalid
    mov w26,w0
.Lq8_map_mask_ready:
    ADDR x0,d1_q8_i8_mask
    str w26,[x0]
    bl getpagesize
    sub x2,x0,#1
    ADDR x0,d1_q8_i8_page_mask
    str x2,[x0]
    mov x0,x20
    ldr x1,=D1_Q8_I8_FILE_BYTES-4096
    bl d1_panel_discard
    cbnz w0,file_error
    ADDR x21,d1_vis_base
    ldr x21,[x21]
    ADDR x22,d1_vis_block_desc
    add x22,x22,#(8*13+2)*32
    ADDR x23,d1_q8_i8_index
    ldr x24,=D1_Q8_I8_FFN_BYTES
    mov w25,#6
.Lq8_map_block:
    .rept 3
    ldr x0,[x22],#32
    add x0,x21,x0
    stp x0,x20,[x23],#16
    add x20,x20,x24
    .endr
    add x22,x22,#10*32
    subs w25,w25,#1
    b.ne .Lq8_map_block
.Lq8_map_done:
    LEAVE
.Lq8_map_mask_invalid:
    ADDR x0,d1_q8_i8_mask_error
    b fatal
.size d1_q8_i8_map,.-d1_q8_i8_map

.type d1_q8_i8_prepare,%function
d1_q8_i8_prepare: /* x0 matrix job with the original Q8_0 activations ready */
    ldr w1,[x0,#56]
    cmp w1,#D1_Q8_I8_POSITIONS
    b.ne .Lq8_prepare_unselected
    ADDR x1,d1_q8_i8_index
    ldr x2,[x1]
    cbz x2,.Lq8_prepare_unselected
    ldr x3,[x0]
    ADDR x6,d1_q8_i8_mask
    ldr w6,[x6]
    mov w4,#D1_Q8_I8_TENSORS
.Lq8_prepare_find:
    ldp x2,x5,[x1],#16
    cmp x2,x3
    b.eq .Lq8_prepare_found
    lsr w6,w6,#1
    subs w4,w4,#1
    b.ne .Lq8_prepare_find
.Lq8_prepare_unselected:
    mov w0,#0
    ret
.Lq8_prepare_found:
    tbz w6,#0,.Lq8_prepare_unselected
    ENTER
    mov x19,x0
    ADDR x1,d1_q8_i8_for_job
    str x5,[x1]
    mov x0,x3
    ldr x1,=D1_Q8_I8_FFN_BYTES
    ADDR x2,d1_q8_i8_page_mask
    ldr x2,[x2]
    bl d1_panel_discard
    cbnz w0,file_error
    ldr x0,[x19,#8]
    ADDR x1,d1_fmla_activations
    ldr x2,[x19,#24]
    ldr w3,[x19,#40]
    lsr w3,w3,#5
    mov w4,#D1_Q8_I8_POSITIONS/4
    bl d1_pack_q8_i8_a4
    ADDR x0,d1_q8_i8_batches
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    ADDR x0,d1_fmla_this_job
    mov w1,#4
    str w1,[x0]
    mov w0,#1
    LEAVE
.size d1_q8_i8_prepare,.-d1_q8_i8_prepare

.macro D1_Q8_PACKED_WORK name,shift,a_bytes,w_bytes,kernel,tail=0
.type \name,%function
\name:
    ENTER
    ADDR x19,d1_batch_job
    add x0,x19,#60
1:  ldaxr w1,[x0]
    add w2,w1,#1
    stlxr w3,w2,[x0]
    cbnz w3,1b
    .if \tail
    sub sp,sp,#272
    ldr w0,[x19,#56]
    str w0,[sp,#256]
    .endif
    ldr w2,[x19,#52]
    lsr w26,w2,#(\shift+2) /* output tiles in a worker's stripe */
    mul w1,w1,w26
    ldr x22,[x19,#16]
    add x22,x22,x1,lsl #(\shift+2)
    ldr x23,[x19,#32]
    ldr w27,[x19,#40]
    lsr w27,w27,#5
    mov w0,#\w_bytes
    mul w24,w27,w0
    mov w0,#\a_bytes
    mul w25,w27,w0
    ADDR x0,d1_q8_i8_for_job
    ldr x21,[x0]
    madd x21,x1,x24,x21
.Lq8_work_tile\@:
    ADDR x19,d1_fmla_activations
    mov x20,x22
    .if \tail
    ldr w28,[sp,#256]
    lsr w28,w28,#2
    .else
    mov w28,#D1_Q8_I8_POSITIONS/4
    .endif
.Lq8_work_positions\@:
    mov x0,x19
    mov x1,x21
    mov x2,x20
    mov x3,x23
    mov w4,w27
    bl \kernel
    add x19,x19,x25
    add x20,x20,x23,lsl #2
    subs w28,w28,#1
    b.ne .Lq8_work_positions\@
    .if \tail
    ldr w28,[sp,#256]
    and w28,w28,#3
    cbz w28,.Lq8_tail_done\@
    mov x0,x19
    mov x1,x21
    mov x2,sp
    mov x3,#64
    mov w4,w27
    bl \kernel
    mov x0,sp
.Lq8_copy_tail\@:
    ldp q0,q1,[x0]
    ldp q2,q3,[x0,#32]
    stp q0,q1,[x20]
    stp q2,q3,[x20,#32]
    add x0,x0,#64
    add x20,x20,x23
    subs w28,w28,#1
    b.ne .Lq8_copy_tail\@
.Lq8_tail_done\@:
    .endif
    add x21,x21,x24
    add x22,x22,#(4<<\shift)
    subs w26,w26,#1
    b.ne .Lq8_work_tile\@
    .if \tail
    add sp,sp,#272
    .endif
    LEAVE
.size \name,.-\name
.endm

D1_Q8_PACKED_WORK d1_q8_i8_batch_work,4,D1_Q8_I8_A_BYTES,D1_Q8_I8_W_BYTES,d1_q8_i8_4x16
D1_Q8_PACKED_WORK d1_q8_ilp_batch_work,1,144,72,d1_q8_ilp_4x2

D1_Q8_PACKED_WORK d1_q8_res_batch_work,4,D1_Q8_I8_A_BYTES,D1_Q8_I8_W_BYTES,d1_q8_i8_resident_4x16,1
    .ifdef D1_Q8_CONV_6_CRC
D1_Q8_PACKED_WORK d1_q8_res_narrow_batch_work,4,D1_Q8_I8_A_BYTES,D1_Q8_I8_W_BYTES,d1_q8_i8_long_4x16,1
    .ifdef D1_Q8_HALF_6_CRC
D1_Q8_PACKED_WORK d1_q8_res_half_batch_work,4,D1_Q8_I8_A_BYTES,D1_Q8_I8_W_BYTES,d1_q8_i8_half_4x16,1
    .ifdef D1_Q8_CERTIFIED_6_CRC
D1_Q8_PACKED_WORK d1_q8_res_certified_batch_work,4,D1_Q8_I8_A_BYTES,D1_Q8_I8_W_BYTES,d1_q8_i8_certified_4x16,1
    .endif
    .endif
    .endif

.type d1_q8_i8_report,%function
d1_q8_i8_report:
    ADDR x0,d1_q8_i8_index
    ldr x0,[x0]
    cbz x0,.Lq8_report_done
    mov w0,#2
    ADDR x1,d1_q8_i8_stats
    ADDR x2,d1_q8_i8_batches
    ldr w2,[x2]
    mov w3,#D1_Q8_I8_POSITIONS
    mul w3,w2,w3
    b dprintf
.Lq8_report_done:
    ret
.size d1_q8_i8_report,.-d1_q8_i8_report
.ltorg

.section .rodata
.balign 8
d1_q8_i8_magic: .ascii "D1Q8I8\001\000"
d1_q8_i8_env: .asciz "D1_VISION_Q8_I8_PANELS"
d1_q8_i8_mask_env: .asciz "D1_VISION_Q8_I8_MASK"
d1_q8_i8_mask_error: .asciz "Expected a nonzero 18-bit packed Q8_0 tensor mask.\n"
d1_q8_i8_stats: .asciz "d1_vision_q8_i8 batches=%u rows=%u\n"
.bss
.balign 8
d1_q8_i8_index: .skip D1_Q8_I8_TENSORS*16
d1_q8_i8_for_job: .skip 8
d1_q8_i8_page_mask: .skip 8
d1_q8_i8_batches: .skip 4
d1_q8_i8_mask: .skip 4
.text
