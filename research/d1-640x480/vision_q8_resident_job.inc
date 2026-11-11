/* D186 panels cover the original early vision weights. Store the ten
 * convolution matrices first so the small schedule can share the file. */
.ifdef D1_Q8_ALL_6_CRC
.ifndef D1_Q8_EARLY_6_CRC
.error "Full resident six-bit mode requires the early six-bit panel CRC"
.endif
.equ D1_Q8_RES_TENSORS,82
.equ D1_Q8_ALL_PAYLOAD,172687360
.equ D1_Q8_ALL_FILE_BYTES,4096+D1_Q8_ALL_PAYLOAD
.ifdef D1_Q8_MERGER_CRC
.equ D1_Q8_MERGER_PAYLOAD,26738688
.equ D1_Q8_MERGER_FILE_BYTES,4096+D1_Q8_MERGER_PAYLOAD
.endif
.else
.equ D1_Q8_RES_TENSORS,46
.endif
.equ D1_Q8_RES_PAYLOAD,99155968
.equ D1_Q8_RES_FILE_BYTES,4096+D1_Q8_RES_PAYLOAD
.ifdef D1_Q8_CONV_6_CRC
.equ D1_Q8_CONV_PAYLOAD,22282240
.equ D1_Q8_CONV_FILE_BYTES,4096+D1_Q8_CONV_PAYLOAD
.endif

.text
.type d1_q8_res_map,%function
d1_q8_res_map:
    ENTER
    ADDR x0,d1_q8_res_env
    bl getenv
    cbz x0,.Lres_map_done
    ldrb w1,[x0]
    cbz w1,.Lres_map_done
    .ifdef D1_Q8_CONV_6_CRC
    mov x24,x0
    ADDR x0,d1_q8_res_compact_env
    bl getenv
    mov w26,#0
    cbz x0,.Lres_pack_mode
    ldrb w26,[x0]
.Lres_pack_mode:
    mov x0,x24
    ldr x25,=D1_Q8_RES_PAYLOAD
    cbz w26,.Lres_original_panel
    .ifdef D1_Q8_ALL_6_CRC
    .ifdef D1_Q8_HALF_6_CRC
    .ifdef D1_Q8_CERTIFIED_6_CRC
    cmp w26,#'s'
    b.ne .Lres_check_half_panel
    ldr x25,=D1_Q8_ALL_PAYLOAD
    ldr x1,=D1_Q8_ALL_FILE_BYTES
    ADDR x2,d1_q8_res_certified_magic6
    b .Lres_bind_panel
.Lres_check_half_panel:
    .endif
    cmp w26,#'h'
    b.ne .Lres_check_all_panel
    ldr x25,=D1_Q8_ALL_PAYLOAD
    ldr x1,=D1_Q8_ALL_FILE_BYTES
    ADDR x2,d1_q8_res_half_magic6
    b .Lres_bind_panel
.Lres_check_all_panel:
    .endif
    cmp w26,#'a'
    b.ne .Lres_check_early_panel
    ldr x25,=D1_Q8_ALL_PAYLOAD
    ldr x1,=D1_Q8_ALL_FILE_BYTES
    ADDR x2,d1_q8_res_all_magic6
    b .Lres_bind_panel
.Lres_check_early_panel:
    .endif
    .ifdef D1_Q8_EARLY_6_CRC
    cmp w26,#'f'
    b.ne .Lres_compact_panel
    ldr x1,=D1_Q8_RES_FILE_BYTES
    ADDR x2,d1_q8_res_early_magic6
    b .Lres_bind_panel
.Lres_compact_panel:
    .endif
    ldr x25,=D1_Q8_CONV_PAYLOAD
    ldr x1,=D1_Q8_CONV_FILE_BYTES
    ADDR x2,d1_q8_res_compact_magic8
    cmp w26,#'8'
    b.eq .Lres_bind_panel
    ADDR x2,d1_q8_res_compact_magic6
    cmp w26,#'6'
    b.ne checksum_error
    b .Lres_bind_panel
.Lres_original_panel:
    .endif
    ldr x1,=D1_Q8_RES_FILE_BYTES
    ADDR x2,d1_q8_res_magic
    .ifdef D1_Q8_CONV_6_CRC
.Lres_bind_panel:
    .endif
    ldr w3,=D1_VISION_CRC
    bl d1_panel_map_checked
    mov x19,x0
    add x20,x19,#4096
    .ifdef D1_PARALLEL_OTHER_PANEL_CRC
    .ifndef D1_Q8_CERTIFIED_6_CRC
    .error "Vision shard checks require the pinned certified sidecar"
    .endif
    .if D1_Q8_CERTIFIED_6_CRC != 0x7c30b13d || D1_Q8_ALL_PAYLOAD != 172687360
    .error "Vision shard CRCs do not bind this sidecar"
    .endif
    cmp w26,#'s'
    b.ne .Lres_serial_crc
    ADDR x0,d1_other_crc_enabled
    ldr w0,[x0]
    cbz w0,.Lres_serial_crc
    ldr w0,[x19,#12]
    ldr w1,=D1_Q8_CERTIFIED_6_CRC
    cmp w0,w1
    b.ne checksum_error
    mov x0,x20
    mov x1,x25
    ldr w2,=0x0749163f
    ldr w3,=0x03af668e
    bl d1_verify_checkpoint_shards
    ADDR x1,d1_other_crc_shards
    ldr w0,[x1]
    add w0,w0,#2
    str w0,[x1]
    ldr w0,=D1_Q8_CERTIFIED_6_CRC
    b .Lres_full_crc
.Lres_serial_crc:
    .endif
    mov x0,x20
    .ifdef D1_Q8_CONV_6_CRC
    mov x1,x25
    .else
    ldr x1,=D1_Q8_RES_PAYLOAD
    .endif
    mov w2,#-1
    bl d1_panel_crc
    mvn w0,w0
    ldr w1,[x19,#12]
    cmp w0,w1
    b.ne checksum_error
    .ifdef D1_PARALLEL_OTHER_PANEL_CRC
.Lres_full_crc:
    .endif
    .ifdef D1_Q8_CONV_6_CRC
    .ifdef D1_Q8_ALL_6_CRC
    .ifdef D1_Q8_HALF_6_CRC
    .ifdef D1_Q8_CERTIFIED_6_CRC
    cmp w26,#'s'
    b.ne .Lres_verify_half
    ldr w1,=D1_Q8_CERTIFIED_6_CRC
    mov w2,#2
    b .Lres_verify_half_crc
.Lres_verify_half:
    .endif
    cmp w26,#'h'
    b.ne .Lres_verify_all
    ldr w1,=D1_Q8_HALF_6_CRC
    mov w2,#1
    .ifdef D1_Q8_CERTIFIED_6_CRC
.Lres_verify_half_crc:
    .endif
    cmp w0,w1
    b.ne checksum_error
    ADDR x1,d1_q8_res_half
    str w2,[x1]
    b .Lres_certified_all
.Lres_verify_all:
    .endif
    cmp w26,#'a'
    b.ne .Lres_verify_compact_or_early
    ldr w1,=D1_Q8_ALL_6_CRC
    cmp w0,w1
    b.ne checksum_error
    .ifdef D1_Q8_HALF_6_CRC
.Lres_certified_all:
    .endif
    ADDR x1,d1_q8_res_all
    mov w2,#1
    str w2,[x1]
    b .Lres_certified_narrow
.Lres_verify_compact_or_early:
    .endif
    ldr w1,=D1_Q8_CONV_6_CRC
    cmp w26,#'6'
    .ifdef D1_Q8_EARLY_6_CRC
    b.eq .Lres_verify_narrow
    cmp w26,#'f'
    b.ne .Lres_pack_verified
    ldr w1,=D1_Q8_EARLY_6_CRC
.Lres_verify_narrow:
    .else
    b.ne .Lres_pack_verified
    .endif
    cmp w0,w1
    b.ne checksum_error
    .ifdef D1_Q8_ALL_6_CRC
.Lres_certified_narrow:
    .endif
    ADDR x1,d1_q8_res_narrow
    mov w0,#1
    str w0,[x1]
.Lres_pack_verified:
    .endif
    bl getpagesize
    sub x2,x0,#1
    ADDR x0,d1_q8_res_page_mask
    str x2,[x0]
    mov x0,x20
    .ifdef D1_Q8_CONV_6_CRC
    mov x1,x25
    .else
    ldr x1,=D1_Q8_RES_PAYLOAD
    .endif
    .ifdef D1_RETAIN_VERIFIED_PAGES
    bl d1_vision_verified_pages
    .else
    bl d1_panel_discard
    .endif
    cbnz w0,file_error
    ADDR x21,d1_vis_base
    ldr x21,[x21]
    ADDR x22,d1_vis_block_desc
    ADDR x23,d1_q8_res_index
    mov w25,#8
    mov w27,#0
.Lres_map_conv:
    ldr w0,[x22,#5*32+16]
    cmp w0,#8
    b.ne .Lres_map_conv_next
    add x0,x22,#5*32
    bl .Lres_map_one
    add x0,x22,#6*32
    bl .Lres_map_one
.Lres_map_conv_next:
    add x22,x22,#13*32
    subs w25,w25,#1
    b.ne .Lres_map_conv
    .ifdef D1_Q8_CONV_6_CRC
    cbz w26,.Lres_map_all
    .ifdef D1_Q8_ALL_6_CRC
    ADDR x0,d1_q8_res_all
    ldr w0,[x0]
    cbnz w0,.Lres_map_certified_full
    .endif
    .ifdef D1_Q8_EARLY_6_CRC
    cmp w26,#'f'
    b.ne .Lres_map_compact_done
    .ifdef D1_Q8_ALL_6_CRC
.Lres_map_certified_full:
    .endif
    mov w26,#0
    b .Lres_map_layers
.Lres_map_compact_done:
    .endif
    ldr x0,=D1_Q8_CONV_FILE_BYTES
    add x0,x19,x0
    cmp x0,x20
    b.ne checksum_error
    b .Lres_map_count
.Lres_map_all:
    .endif
    ADDR x0,d1_q8_res_conv_only_env
    bl getenv
    cbz x0,.Lres_map_rest
    ldrb w0,[x0]
    cmp w0,#'1'
    b.eq .Lres_map_count
.Lres_map_rest:
    mov w26,#0
    ADDR x0,d1_q8_res_subset_env
    bl getenv
    cbz x0,.Lres_map_layers
    ldrb w1,[x0]
    cbz w1,.Lres_map_layers
    mov x24,x0
    ADDR x1,d1_q8_res_subset_ffn
    bl strcmp
    mov w26,#1
    cbz w0,.Lres_map_layers
    mov x0,x24
    ADDR x1,d1_q8_res_subset_attn
    bl strcmp
    mov w26,#2
    cbz w0,.Lres_map_layers
    ADDR x0,d1_q8_res_subset_error
    b fatal
.Lres_map_layers:
    ADDR x22,d1_vis_block_desc
    mov w25,#8
.Lres_map_layer:
    cmp w26,#2
    cset w27,eq /* attention-only subset skips FFN panels */
    add x0,x22,#2*32
    bl .Lres_map_one
    add x0,x22,#3*32
    bl .Lres_map_one
    add x0,x22,#4*32
    bl .Lres_map_one
    ldr w0,[x22,#7*32+16]
    cmp w0,#8
    b.ne .Lres_map_next
    cmp w26,#1
    cset w27,eq /* FFN-only subset skips attention panels */
    add x0,x22,#7*32
    bl .Lres_map_one
    add x0,x22,#8*32
    bl .Lres_map_one
    add x0,x22,#9*32
    bl .Lres_map_one
    add x0,x22,#10*32
    bl .Lres_map_one
.Lres_map_next:
    add x22,x22,#13*32
    subs w25,w25,#1
    b.ne .Lres_map_layer
    .ifdef D1_Q8_ALL_6_CRC
    ADDR x0,d1_q8_res_all
    ldr w0,[x0]
    cbnz w0,.Lres_map_late_start
    .endif
    ldr x0,=D1_Q8_RES_FILE_BYTES
    add x0,x19,x0
    cmp x0,x20
    b.ne checksum_error
    .ifdef D1_Q8_ALL_6_CRC
    b .Lres_map_count
.Lres_map_late_start:
    mov w25,#6
    mov w27,#0
.Lres_map_late:
    add x0,x22,#2*32
    bl .Lres_map_one
    add x0,x22,#3*32
    bl .Lres_map_one
    add x0,x22,#4*32
    bl .Lres_map_one
    ldr w0,[x22,#5*32+16]
    cmp w0,#8
    b.ne .Lres_map_late_attention
    add x0,x22,#5*32
    bl .Lres_map_one
    add x0,x22,#6*32
    bl .Lres_map_one
    b .Lres_map_late_next
.Lres_map_late_attention:
    add x0,x22,#7*32
    bl .Lres_map_one
    add x0,x22,#8*32
    bl .Lres_map_one
    add x0,x22,#9*32
    bl .Lres_map_one
    add x0,x22,#10*32
    bl .Lres_map_one
.Lres_map_late_next:
    add x22,x22,#13*32
    subs w25,w25,#1
    b.ne .Lres_map_late
    ldr x0,=D1_Q8_ALL_FILE_BYTES
    add x0,x19,x0
    cmp x0,x20
    b.ne checksum_error
    .endif
.Lres_map_count:
    .ifdef D1_Q8_MERGER_CRC
    ADDR x0,d1_q8_merger_env
    bl getenv
    cbz x0,.Lres_store_count
    ldrb w1,[x0]
    cbz w1,.Lres_store_count
    ADDR x1,d1_q8_res_all
    ldr w1,[x1]
    cbz w1,checksum_error
    ldr x1,=D1_Q8_MERGER_FILE_BYTES
    ADDR x2,d1_q8_merger_magic
    ldr w3,=D1_VISION_CRC
    bl d1_panel_map_checked
    mov x19,x0
    add x20,x19,#4096
    mov x0,x20
    ldr x1,=D1_Q8_MERGER_PAYLOAD
    mov w2,#-1
    bl d1_panel_crc
    mvn w0,w0
    ldr w1,[x19,#12]
    cmp w0,w1
    b.ne checksum_error
    ldr w1,=D1_Q8_MERGER_CRC
    cmp w0,w1
    b.ne checksum_error
    mov x0,x20
    ldr x1,=D1_Q8_MERGER_PAYLOAD
    ADDR x2,d1_q8_res_page_mask
    ldr x2,[x2]
    .ifdef D1_RETAIN_VERIFIED_PAGES
    bl d1_vision_verified_pages
    .else
    bl d1_panel_discard
    .endif
    cbnz w0,file_error
    ADDR x0,d1_q8_merger_pack_base
    str x20,[x0]
    mov w27,#0
    ADDR x0,d1_vis_up_desc
    bl .Lres_map_one
    ADDR x0,d1_vis_down_desc
    bl .Lres_map_one
    ldr x0,=D1_Q8_MERGER_FILE_BYTES
    add x0,x19,x0
    cmp x0,x20
    b.ne checksum_error
.Lres_store_count:
    .endif
    ADDR x0,d1_q8_res_index
    sub x1,x23,x0
    lsr x1,x1,#4
    ADDR x0,d1_q8_res_count
    str w1,[x0]
.Lres_map_done:
    LEAVE

.Lres_map_one: /* x0 descriptor, x20 payload cursor, x21 projector, x23 index cursor */
    cbnz w27,.Lres_map_skip_index
    ldr x1,[x0]
    add x1,x21,x1
    stp x1,x20,[x23],#16
.Lres_map_skip_index:
    ldr x1,[x0,#24]
    add x20,x20,x1
    ret
.size d1_q8_res_map,.-d1_q8_res_map

.type d1_q8_res_prepare,%function
d1_q8_res_prepare: /* x0 early vision matrix job with raw Q8_0 activations */
    .ifdef D1_Q8_RES_REUSE
    mov w6,#0
    b .Lres_prepare_start
d1_q8_res_prepare_reused: /* the previous job has the same interleaved activations */
    mov w6,#1
.Lres_prepare_start:
    .endif
    ADDR x1,d1_q8_res_count
    ldr w4,[x1]
    cbz w4,.Lres_unselected
    ldr w1,[x0,#56]
    cmp w1,#4
    b.lo .Lres_unselected
    cmp w1,#D1_FMLA_MAX_POS
    b.hi .Lres_unselected
    ADDR x1,d1_q8_res_index
    ldr x3,[x0]
.Lres_find:
    ldp x2,x5,[x1],#16
    cmp x2,x3
    b.eq .Lres_selected
    subs w4,w4,#1
    b.ne .Lres_find
.Lres_unselected:
    mov w0,#0
    ret
.Lres_selected:
    ENTER
    .ifdef D1_Q8_RES_REUSE
    mov w28,w6
    .endif
    mov x19,x0
    ldr w21,[x19,#56]
    add w22,w21,#3
    and w22,w22,#~3
    sub w23,w22,w21
    ldr x24,[x19,#24] /* raw packed stride */
    mul x0,x22,x24
    ldr x1,=d1_batch_pack_end-d1_batch_pack
    cmp x0,x1
    b.hi .Lres_cannot_pack
    ADDR x1,d1_q8_i8_for_job
    str x5,[x1]
    ldp w1,w2,[x19,#48]
    mul w1,w1,w2
    mov x0,x3
    ADDR x2,d1_q8_res_page_mask
    ldr x2,[x2]
    bl d1_panel_discard
    cbnz w0,file_error
    .ifdef D1_Q8_RES_REUSE
    cbnz w28,.Lres_publish
    .endif
    ldr x20,[x19,#8]
    cbz w23,.Lres_pack
    madd x0,x21,x24,x20
    mov w1,#0
    mul x2,x23,x24
    bl memset
.Lres_pack:
    mov x0,x20
    ADDR x1,d1_fmla_activations
    mov x2,x24
    ldr w3,[x19,#40]
    lsr w3,w3,#5
    lsr w4,w22,#2
    bl d1_pack_q8_i8_a4
    .ifdef D1_Q8_RES_REUSE
.Lres_publish:
    .endif
    ADDR x0,d1_q8_res_counters
    ldp w1,w2,[x0]
    ldr w3,[x0,#8]
    add w1,w1,#1
    add w2,w2,w21
    add w3,w3,w23
    stp w1,w2,[x0]
    str w3,[x0,#8]
    ADDR x0,d1_fmla_this_job
    mov w1,#6
    .ifdef D1_Q8_CONV_6_CRC
    ADDR x2,d1_q8_res_narrow
    ldr w2,[x2]
    add w1,w1,w2
    .ifdef D1_Q8_HALF_6_CRC
    ADDR x2,d1_q8_res_half
    ldr w2,[x2]
    add w1,w1,w2
    .endif
    .endif
    .ifdef D1_Q8_MERGER_CRC
    ADDR x2,d1_q8_merger_pack_base
    ldr x2,[x2]
    cbz x2,.Lres_store_job_kind
    ADDR x3,d1_q8_i8_for_job
    ldr x3,[x3]
    sub x3,x3,x2
    ldr x2,=D1_Q8_MERGER_PAYLOAD
    cmp x3,x2
    mov w2,#9
    csel w1,w2,w1,lo
.Lres_store_job_kind:
    .endif
    str w1,[x0]
    mov w0,#1
    LEAVE
.Lres_cannot_pack:
    mov w0,#0
    LEAVE
.size d1_q8_res_prepare,.-d1_q8_res_prepare

.type d1_q8_res_report,%function
d1_q8_res_report:
    ADDR x0,d1_q8_res_count
    ldr w0,[x0]
    cbz w0,.Lres_report_done
    mov w0,#2
    ADDR x1,d1_q8_res_stats
    ADDR x2,d1_q8_res_counters
    ldr w4,[x2,#8]
    ldp w2,w3,[x2]
    b dprintf
.Lres_report_done:
    ret
.size d1_q8_res_report,.-d1_q8_res_report
.ltorg

.section .rodata
.balign 8
d1_q8_res_magic: .ascii "D1Q8RS\001\000"
    .ifdef D1_Q8_CONV_6_CRC
    .balign 8
d1_q8_res_compact_magic8: .ascii "D1Q8C8\001\000"
d1_q8_res_compact_magic6: .ascii "D1Q8C6\001\000"
    .ifdef D1_Q8_EARLY_6_CRC
d1_q8_res_early_magic6: .ascii "D1Q8E6\001\000"
    .endif
    .ifdef D1_Q8_ALL_6_CRC
d1_q8_res_all_magic6: .ascii "D1Q8A6\001\000"
    .ifdef D1_Q8_HALF_6_CRC
d1_q8_res_half_magic6: .ascii "D1Q8H6\001\000"
    .ifdef D1_Q8_CERTIFIED_6_CRC
d1_q8_res_certified_magic6: .ascii "D1Q8S6\001\000"
    .endif
    .endif
    .endif
d1_q8_res_compact_env: .asciz "D1_VISION_Q8_CONV_COMPACT"
    .endif
d1_q8_res_env: .asciz "D1_VISION_Q8_RESIDENT_PANELS"
.ifdef D1_Q8_MERGER_CRC
.balign 8
d1_q8_merger_magic: .ascii "D1Q8M6\001\000"
d1_q8_merger_env: .asciz "D1_VISION_MERGER_PANELS"
.endif
d1_q8_res_conv_only_env: .asciz "D1_VISION_Q8_RESIDENT_CONV_ONLY"
d1_q8_res_subset_env: .asciz "D1_VISION_Q8_RESIDENT_SUBSET"
d1_q8_res_subset_ffn: .asciz "ffn"
d1_q8_res_subset_attn: .asciz "attn"
d1_q8_res_subset_error: .asciz "Invalid early Q8_0 subset. Expected ffn or attn.\n"
d1_q8_res_stats: .asciz "d1_vision_q8_resident batches=%u rows=%u padded=%u\n"
    .ifdef D1_Q8_RES_REUSE
d1_q8_res_reuse_stats: .asciz "d1_vision_resident_pack_reuse=%u\n"
    .endif
.bss
.balign 8
d1_q8_res_index: .skip D1_Q8_RES_TENSORS*16
.ifdef D1_Q8_MERGER_CRC
.skip 2*16
d1_q8_merger_pack_base: .skip 8
.endif
d1_q8_res_page_mask: .skip 8
d1_q8_res_count: .skip 4
d1_q8_res_counters: .skip 12
    .ifdef D1_Q8_RES_REUSE
d1_q8_res_reuse_count: .skip 4
    .endif
    .ifdef D1_Q8_CONV_6_CRC
d1_q8_res_narrow: .skip 4
    .endif
    .ifdef D1_Q8_ALL_6_CRC
d1_q8_res_all: .skip 4
    .ifdef D1_Q8_HALF_6_CRC
d1_q8_res_half: .skip 4
    .endif
    .endif
.text
