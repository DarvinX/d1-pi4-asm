/* Exact D176 panels. Original tensor pages are released only when a
 * matched batched projection selects the packed route. */
.equ D1_Q4_I8_FILE_BYTES,4096+735330304
.equ D1_Q4_I8_TENSORS,125
.equ D1_Q4_I8_INDEX_BYTES,48
.equ D1_Q4_I8_W_BYTES,2368
.equ D1_Q4_I8_A_BYTES,1104
.equ D1_Q4_I8_CONV_RAW_PAGES,2048*(2048/256)*144/4096
.equ D1_Q4_I8_CONV_PACKED_PAGES,(2048/16)*(2048/256)*D1_Q4_I8_W_BYTES/4096

.text
.type d1_q4_i8_map,%function
d1_q4_i8_map:
    ENTER
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    sub sp,sp,#16
    str wzr,[sp]
    .endif
    ADDR x0,d1_q4_i8_env
    bl getenv
    cbz x0,.Lq4_map_done
    ldrb w1,[x0]
    cbz w1,.Lq4_map_done
    ldr x1,=D1_Q4_I8_FILE_BYTES
    ADDR x2,d1_q4_i8_magic
    bl d1_panel_map_file
    mov x20,x0
    ADDR x0,d1_q4_i8_min_env
    bl getenv
    mov w19,#128
    cbz x0,.Lq4_map_threshold
    mov x21,x0
    ADDR x1,d1_q4_i8_min48
    bl strcmp
    cbnz w0,.Lq4_map_test128
    mov w19,#48
    b .Lq4_map_threshold
.Lq4_map_test128:
    .ifdef D1_SHORT_PACKED_ROWS
    mov x0,x21
    ADDR x1,d1_q4_i8_min28
    bl strcmp
    cbnz w0,.Lq4_map_legacy128
    mov w19,#28
    b .Lq4_map_threshold
.Lq4_map_legacy128:
    .endif
    mov x0,x21
    ADDR x1,d1_q4_i8_min128
    bl strcmp
    cbnz w0,.Lq4_invalid
.Lq4_map_threshold:
    ADDR x0,d1_q4_i8_min
    str w19,[x0]
    ADDR x0,d1_q4_i8_slices_env
    bl getenv
    cbz x0,.Lq4_map_page_size
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_q4_i8_slices_enabled
    str w0,[x1]
.Lq4_map_page_size:
    bl getpagesize
    sub x23,x0,#1
    ADDR x0,d1_q4_i8_page_mask
    str x23,[x0]
    .ifdef D1_Q4_SEED_PRODUCT
    ADDR x0,d1_q4_seed_env
    bl getenv
    cbz x0,.Lq4_seed_configured
    ldrb w1,[x0]
    cmp w1,#'1'
    cset w1,eq
    ADDR x0,d1_q4_seed_enabled
    str w1,[x0]
.Lq4_seed_configured:
    .endif
    .ifdef D1_Q4_GATE_UP_REUSE
    ADDR x0,d1_q4_gate_up_reuse_env
    bl getenv
    cbz x0,.Lq4_reuse_configured
    ldrb w1,[x0]
    cmp w1,#'1'
    cset w1,eq
    ADDR x0,d1_q4_gate_up_reuse_enabled
    str w1,[x0]
.Lq4_reuse_configured:
    .endif
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    ADDR x0,d1_q4_panel_shards_env
    bl getenv
    cbz x0,.Lq4_map_index
    ldrb w0,[x0]
    cmp w0,#'1'
    b.ne .Lq4_map_index
    ldr w0,[x20,#12]
    ldr w1,=0xec0d3a6f
    cmp w0,w1
    b.ne .Lq4_invalid
    add x0,x20,#4096
    ldr x1,=D1_Q4_I8_FILE_BYTES-4096
    ldr w2,=0x3c194691
    ldr w3,=0xa0587c97
    mov x4,x23
    .ifdef D1_RETAIN_VERIFIED_PAGES
    ADDR x5,d1_retain_pages_enabled
    ldr w5,[x5]
    cbz w5,.Lq4_verified_pages_mode
    ADDR x5,d1_retain_pages_counts
    str w1,[x5]
    mov x4,#0
.Lq4_verified_pages_mode:
    .endif
    bl d1_verify_bounded_shards
    mov w0,#2
    str w0,[sp]
    ADDR x1,d1_q4_panel_shards
    str w0,[x1]
.Lq4_map_index:
    .endif
    ADDR x19,model_base
    ldr x19,[x19]
    ADDR x21,layer_tensors
    mov w22,#LAYERS*18
    ADDR x24,d1_q4_i8_index
    add x25,x20,#4096
    mov w26,#-1
    mov w27,#0
.Lq4_map_tensor:
    ldr w0,[x21,#16]
    cmp w0,#12
    b.ne .Lq4_map_next
    ldp w0,w1,[x21,#8]
    cbz w0,.Lq4_map_next
    cbz w1,.Lq4_map_next
    tst w0,#255
    b.ne .Lq4_map_next
    tst w1,#63
    b.ne .Lq4_map_next
    cmp w27,#D1_Q4_I8_TENSORS
    b.hs .Lq4_invalid
    stp w0,w1,[x24,#32]
    lsr w0,w0,#8
    lsr w1,w1,#4
    mul w0,w0,w1
    mov w1,#D1_Q4_I8_W_BYTES
    mul w28,w0,w1
    ldr x0,[x21]
    ldr x1,[x21,#24]
    add x0,x19,x0
    stp x0,x25,[x24]
    stp x1,xzr,[x24,#16]
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    ldr w0,[sp]
    cbnz w0,.Lq4_map_index_next
    .endif
    mov x0,x25
    mov x1,x28
    mov w2,w26
    bl d1_panel_crc
    mov w26,w0
    /* Keep validation RSS bounded without evicting the OS file cache. */
    mov x0,x25
    mov x1,x28
    mov x2,x23
    bl d1_panel_discard
    cbnz w0,file_error
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
.Lq4_map_index_next:
    .endif
    add x25,x25,x28
    add x24,x24,#D1_Q4_I8_INDEX_BYTES
    add w27,w27,#1
.Lq4_map_next:
    add x21,x21,#32
    subs w22,w22,#1
    b.ne .Lq4_map_tensor
    cmp w27,#D1_Q4_I8_TENSORS
    b.ne .Lq4_invalid
    ldr x0,=D1_Q4_I8_FILE_BYTES
    add x0,x20,x0
    cmp x0,x25
    b.ne .Lq4_invalid
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    ldr w0,[sp]
    cbnz w0,.Lq4_map_done
    .endif
    mvn w26,w26
    ldr w0,[x20,#12]
    cmp w0,w26
    b.ne .Lq4_invalid
.Lq4_map_done:
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    add sp,sp,#16
    .endif
    LEAVE
.Lq4_invalid:
    ADDR x0,d1_q4_i8_error
    b fatal
.size d1_q4_i8_map,.-d1_q4_i8_map

.type d1_q4_i8_find,%function
d1_q4_i8_find: /* x0 original weights, w1 width, w2 rows -> packed or NULL */
    ENTER
    ADDR x19,d1_q4_i8_index
    mov w4,#2048
    cmp w1,w4
    ccmp w2,w4,#0,eq
    b.ne .Lq4_find_full
    ADDR x3,d1_q4_i8_slices_enabled
    ldr w3,[x3]
    cbnz w3,.Lq4_find_with_slices
.Lq4_find_full:
    mov w3,#D1_Q4_I8_TENSORS
.Lq4_find:
    ldr x4,[x19]
    cmp x0,x4
    b.eq .Lq4_found
    add x19,x19,#D1_Q4_I8_INDEX_BYTES
    subs w3,w3,#1
    b.ne .Lq4_find
    b .Lq4_not_found
.Lq4_found:
    ldp w3,w4,[x19,#32]
    cmp w1,w3
    b.ne .Lq4_not_found
    cmp w2,w4
    b.ne .Lq4_not_found
    ldr x3,[x19,#24]
    cbnz x3,.Lq4_selected
    ldr x1,[x19,#16]
    ADDR x2,d1_q4_i8_page_mask
    ldr x2,[x2]
    bl d1_panel_discard
    cbnz w0,file_error
    mov x0,#1
    str x0,[x19,#24]
.Lq4_selected:
    ldr x0,[x19,#8]
    LEAVE
.Lq4_find_with_slices:
    mov w3,#D1_Q4_I8_TENSORS
    mov w8,#3*2048
.Lq4_slice_next:
    ldr x4,[x19]
    ldp w5,w6,[x19,#32]
    cmp w5,#2048
    b.ne .Lq4_slice_advance
    cmp w6,w8
    b.eq .Lq4_try_third
    cmp x0,x4
    b.eq .Lq4_found
    b .Lq4_slice_advance
.Lq4_try_third:
    sub x7,x0,x4
    cmp x7,#0
    mov x20,#0
    b.eq .Lq4_slice_found
    cmp x7,#D1_Q4_I8_CONV_RAW_PAGES,lsl #12
    mov x20,#D1_Q4_I8_CONV_PACKED_PAGES*4096
    b.eq .Lq4_slice_found
    cmp x7,#2*D1_Q4_I8_CONV_RAW_PAGES,lsl #12
    add x20,x20,x20
    b.eq .Lq4_slice_found
.Lq4_slice_advance:
    add x19,x19,#D1_Q4_I8_INDEX_BYTES
    subs w3,w3,#1
    b.ne .Lq4_slice_next
    b .Lq4_not_found
.Lq4_slice_found:
    /* The three equal output ranges start on packed-panel boundaries. */
    ldr x3,[x19,#8]
    add x20,x20,x3
    mov x1,#D1_Q4_I8_CONV_RAW_PAGES*4096
    ADDR x2,d1_q4_i8_page_mask
    ldr x2,[x2]
    bl d1_panel_discard
    cbnz w0,file_error
    ADDR x1,d1_q4_i8_slice_count
    ldr w0,[x1]
    add w0,w0,#1
    str w0,[x1]
    mov x0,x20
    LEAVE
.Lq4_not_found:
    mov x0,#0
    LEAVE
.size d1_q4_i8_find,.-d1_q4_i8_find

.type d1_q4_i8_prepare,%function
d1_q4_i8_prepare: /* x0 batch job with original Q8_K activations ready */
    .ifdef D1_Q4_GATE_UP_REUSE
    mov w1,#0
    b .Lq4_prepare_entry
.type d1_q4_i8_prepare_reused,%function
d1_q4_i8_prepare_reused:
    mov w1,#1
.Lq4_prepare_entry:
    .endif
    ENTER
    mov x19,x0
    .ifdef D1_Q4_GATE_UP_REUSE
    mov w25,w1
    .endif
    ADDR x0,d1_q4_i8_min
    ldr w0,[x0]
    cbz w0,.Lq4_prepare_done
    ldr w26,[x19,#56]
    cmp w26,w0
    b.lo .Lq4_prepare_done
    cmp w26,#CTX
    b.hi .Lq4_prepare_done
    add w20,w26,#3
    and w20,w20,#~3
    ldr x21,[x19,#8]
    ldr x22,[x19,#24]
    mul x0,x20,x22
    ldr x1,=d1_batch_pack_end-d1_batch_pack
    cmp x0,x1
    b.hi .Lq4_prepare_done
    ldr x0,[x19]
    ldr w1,[x19,#40]
    ldr w2,[x19,#52]
    lsr w23,w1,#8
    bl d1_q4_i8_find
    cbz x0,.Lq4_prepare_done
    ADDR x1,d1_q4_i8_for_job
    str x0,[x1]
    .ifdef D1_Q4_SEED_PRODUCT
    ADDR x2,d1_q4_i8_exact_4x16
    ADDR x1,d1_q4_seed_enabled
    ldr w1,[x1]
    cbz w1,.Lq4_seed_kernel_ready
    cmp w23,#8 /* measured 2048-channel tile */
    b.ne .Lq4_seed_kernel_ready
    cmp w26,#64
    b.hi .Lq4_seed_kernel_ready
    ADDR x2,d1_q4_i8_seeded_4x16
    ADDR x1,d1_q4_seed_jobs
    ldr w3,[x1]
    add w3,w3,#1
    str w3,[x1]
.Lq4_seed_kernel_ready:
    ADDR x1,d1_q4_kernel_for_job
    str x2,[x1]
    .endif
    sub w27,w20,w26
    .ifdef D1_Q4_GATE_UP_REUSE
    cbz w25,.Lq4_pack_required
    ADDR x0,d1_q4_gate_up_reuse_counts
    ldp w1,w2,[x0]
    add w1,w1,#1
    add w2,w2,w26
    stp w1,w2,[x0]
    b .Lq4_finalize_job
.Lq4_pack_required:
    .endif
    cbz w27,.Lq4_prepare_pack
    madd x0,x26,x22,x21
    mov w1,#0
    mul x2,x27,x22
    bl memset
.Lq4_prepare_pack:
    ADDR x24,d1_fmla_activations
    mov x25,x21
    mov w28,w20
.Lq4_pack_positions:
    mov x19,x25
    mov w20,w23
.Lq4_pack_blocks:
    mov x0,x19
    mov x1,x24
    mov x2,x22
    bl d1_q4_i8_pack_a4
    add x19,x19,#320
    add x24,x24,#D1_Q4_I8_A_BYTES
    subs w20,w20,#1
    b.ne .Lq4_pack_blocks
    add x25,x25,x22,lsl #2
    subs w28,w28,#4
    b.ne .Lq4_pack_positions
    .ifdef D1_Q4_GATE_UP_REUSE
.Lq4_finalize_job:
    .endif
    ADDR x0,d1_q4_i8_counts
    ldp w1,w2,[x0]
    ldr w3,[x0,#8]
    add w1,w1,#1
    add w2,w2,w26
    add w3,w3,w27
    stp w1,w2,[x0]
    str w3,[x0,#8]
    ADDR x0,d1_fmla_this_job
    mov w1,#3
    str w1,[x0]
.Lq4_prepare_done:
    LEAVE
.size d1_q4_i8_prepare,.-d1_q4_i8_prepare
.ifdef D1_Q4_GATE_UP_REUSE
.size d1_q4_i8_prepare_reused,.-d1_q4_i8_prepare_reused
.endif

.type d1_q4_i8_batch_work,%function
d1_q4_i8_batch_work:
    ENTER
    sub sp,sp,#288
    ADDR x19,d1_batch_job
    add x0,x19,#60
1:  ldaxr w1,[x0]
    add w2,w1,#1
    stlxr w3,w2,[x0]
    cbnz w3,1b
    ldr w2,[x19,#52]
    lsr w26,w2,#6 /* output tiles per worker */
    mul w1,w1,w26
    ldr x27,[x19,#16]
    add x27,x27,x1,lsl #6
    ldr x22,[x19,#32]
    ldr w23,[x19,#56]
    ldr w24,[x19,#40]
    lsr w24,w24,#8
    mov w0,#D1_Q4_I8_W_BYTES
    mul w25,w24,w0
    mov w0,#D1_Q4_I8_A_BYTES
    mul w28,w24,w0
    ADDR x0,d1_q4_i8_for_job
    ldr x21,[x0]
    madd x21,x1,x25,x21
    ADDR x20,d1_fmla_activations
    .ifdef D1_Q4_SEED_PRODUCT
    ADDR x19,d1_q4_kernel_for_job
    ldr x19,[x19]
    .endif
.Lq4_work_rows:
    stp x20,x27,[sp,#256]
    str w23,[sp,#272]
.Lq4_work_positions:
    ldp x0,x2,[sp,#256]
    mov x1,x21
    mov x3,x22
    mov w4,w24
    ldr w5,[sp,#272]
    cmp w5,#4
    b.lo .Lq4_work_tail
    .ifdef D1_Q4_SEED_PRODUCT
    blr x19
    .else
    bl d1_q4_i8_exact_4x16
    .endif
    ldp x0,x2,[sp,#256]
    add x0,x0,x28
    add x2,x2,x22,lsl #2
    stp x0,x2,[sp,#256]
    ldr w5,[sp,#272]
    subs w5,w5,#4
    str w5,[sp,#272]
    b.ne .Lq4_work_positions
    b .Lq4_work_next
.Lq4_work_tail:
    mov x2,sp
    mov x3,#64
    .ifdef D1_Q4_SEED_PRODUCT
    blr x19
    .else
    bl d1_q4_i8_exact_4x16
    .endif
    ldr x2,[sp,#264]
    ldr w5,[sp,#272]
    mov x0,sp
.Lq4_copy_tail:
    ldp q0,q1,[x0]
    ldp q2,q3,[x0,#32]
    stp q0,q1,[x2]
    stp q2,q3,[x2,#32]
    add x0,x0,#64
    add x2,x2,x22
    subs w5,w5,#1
    b.ne .Lq4_copy_tail
.Lq4_work_next:
    add x21,x21,x25
    add x27,x27,#64
    subs w26,w26,#1
    b.ne .Lq4_work_rows
    add sp,sp,#288
    LEAVE
.size d1_q4_i8_batch_work,.-d1_q4_i8_batch_work

.type d1_q4_i8_report,%function
d1_q4_i8_report:
    ADDR x0,d1_q4_i8_min
    ldr w0,[x0]
    cbz w0,.Lq4_report_done
    mov w0,#2
    ADDR x1,d1_q4_i8_stats
    ADDR x2,d1_q4_i8_counts
    ldr w4,[x2,#8]
    ldp w2,w3,[x2]
    b dprintf
.Lq4_report_done:
    ret
.size d1_q4_i8_report,.-d1_q4_i8_report

.type d1_q4_i8_slice_report,%function
d1_q4_i8_slice_report:
    ADDR x0,d1_q4_i8_slices_enabled
    ldr w0,[x0]
    cbz w0,.Lq4_slice_report_done
    mov w0,#2
    ADDR x1,d1_q4_i8_slice_stats
    ADDR x2,d1_q4_i8_slice_count
    ldr w2,[x2]
    b dprintf
.Lq4_slice_report_done:
    ret
.size d1_q4_i8_slice_report,.-d1_q4_i8_slice_report
.ltorg

.section .rodata
.balign 8
d1_q4_i8_magic: .ascii "D1Q4I8\001\000"
d1_q4_i8_env: .asciz "D1_Q4_I8_PANELS"
d1_q4_i8_min_env: .asciz "D1_Q4_I8_MIN_ROWS"
d1_q4_i8_slices_env: .asciz "D1_Q4_I8_CONV_SLICES"
    .ifdef D1_SHORT_PACKED_ROWS
d1_q4_i8_min28: .asciz "28"
    .endif
d1_q4_i8_min48: .asciz "48"
d1_q4_i8_min128: .asciz "128"
d1_q4_i8_error: .asciz "Invalid checkpoint-bound Q4_K panel file or row threshold.\n"
d1_q4_i8_stats: .asciz "d1_q4_i8 batches=%u rows=%u padded=%u\n"
d1_q4_i8_slice_stats: .asciz "d1_q4_i8_conv_slices=%u\n"
.ifdef D1_Q4_SEED_PRODUCT
d1_q4_seed_env: .asciz "D1_Q4_SEED_PRODUCT"
d1_q4_seed_stats: .asciz "d1_q4_seed_jobs=%u\n"
.endif
.ifdef D1_Q4_GATE_UP_REUSE
d1_q4_gate_up_reuse_env: .asciz "D1_Q4_GATE_UP_REUSE"
d1_q4_gate_up_reuse_stats: .asciz "d1_q4_gate_up_reuse jobs=%u rows=%u\n"
.endif
.ifdef D1_PARALLEL_Q4_PANEL_CRC
d1_q4_panel_shards_env: .asciz "D1_PARALLEL_Q4_PANEL_CRC"
d1_q4_panel_shards_stats: .asciz "d1_q4_panel_crc_shards=%u\n"
.endif
.bss
.balign 8
d1_q4_i8_index: .skip D1_Q4_I8_TENSORS*D1_Q4_I8_INDEX_BYTES
d1_q4_i8_page_mask: .skip 8
d1_q4_i8_for_job: .skip 8
.ifdef D1_Q4_SEED_PRODUCT
d1_q4_kernel_for_job: .skip 8
d1_q4_seed_jobs: .skip 4
d1_q4_seed_enabled: .skip 4
.endif
.ifdef D1_Q4_GATE_UP_REUSE
d1_q4_gate_up_reuse_enabled: .skip 4
d1_q4_gate_up_reuse_counts: .skip 8
.endif
d1_q4_i8_min: .skip 4
d1_q4_i8_counts: .skip 12
d1_q4_i8_slices_enabled: .skip 4
d1_q4_i8_slice_count: .skip 4
.ifdef D1_PARALLEL_Q4_PANEL_CRC
d1_q4_panel_shards: .skip 4
.endif
.text
