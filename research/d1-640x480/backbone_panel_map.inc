/* Common read-only sidecar mapping and checked model binding. */
.text
.type d1_panel_map_file,%function
d1_panel_map_file: /* x0 path, x1 size, x2 eight-byte magic */
    ldr w3,=MODEL_CRC
d1_panel_map_checked: /* same arguments, w3 expected source CRC */
    ENTER
    mov w23,w3
    mov x19,x1
    ldr x20,[x2]
    mov w1,#0
    bl open
    cmp w0,#0
    b.lt file_error
    mov w21,w0
    mov x1,#0
    mov w2,#2
    bl lseek
    cmp x0,x19
    b.ne .Lpanel_invalid
    mov x0,#0
    mov x1,x19
    mov w2,#1
    mov w3,#2
    mov w4,w21
    mov x5,#0
    bl mmap
    cmn x0,#1
    b.eq file_error
    mov x22,x0
    mov w0,w21
    bl close
    ldr x0,[x22]
    cmp x0,x20
    b.ne .Lpanel_invalid
    ldr w0,[x22,#8]
    cmp w0,w23
    b.ne .Lpanel_invalid
    mov x0,x22
    LEAVE
.Lpanel_invalid:
    ADDR x0,d1_panel_header_error
    b fatal
.size d1_panel_map_file,.-d1_panel_map_file

.type d1_panel_discard,%function
d1_panel_discard: /* x0 address, x1 bytes, x2 page mask */
    add x1,x0,x1
    add x0,x0,x2
    bic x0,x0,x2
    bic x1,x1,x2
    subs x1,x1,x0
    b.ls .Lpanel_no_pages
    mov w2,#4 /* MADV_DONTNEED */
    b madvise
.Lpanel_no_pages:
    mov w0,#0
    ret
.size d1_panel_discard,.-d1_panel_discard

.ifdef D1_RETAIN_VERIFIED_PAGES
.ifndef D1_DEFER_MODEL_CRC
.error "Retaining panel pages requires the deferred backbone loader"
.endif
.type d1_read_retained_pages,%function
d1_read_retained_pages:
    ENTER
    ADDR x0,d1_retain_pages_env
    bl getenv
    cbz x0,.Lretain_pages_done
    ldrb w1,[x0]
    cmp w1,#'1'
    b.ne .Lretain_pages_done
    ADDR x0,d1_defer_enabled
    ldr w1,[x0]
    cbz w1,d1_decision_invalid
    ADDR x0,d1_retain_pages_enabled
    str w1,[x0]
.Lretain_pages_done:
    LEAVE
.size d1_read_retained_pages,.-d1_read_retained_pages

.type d1_vision_verified_pages,%function
d1_vision_verified_pages: /* x0 verified mapping, x1 bytes, x2 page mask */
    ADDR x3,d1_retain_pages_enabled
    ldr w3,[x3]
    cbz w3,d1_panel_discard
    ADDR x3,d1_retain_pages_counts
    ldr w4,[x3,#4]
    add w4,w4,w1
    str w4,[x3,#4]
    mov w0,#0
    ret
.size d1_vision_verified_pages,.-d1_vision_verified_pages
.section .rodata
d1_retain_pages_env: .asciz "D1_RETAIN_VERIFIED_PAGES"
d1_retain_pages_stats: .asciz "d1_retained_panel_bytes q4=%u vision=%u\n"
.bss
.balign 4
d1_retain_pages_enabled: .skip 4
d1_retain_pages_counts: .skip 8
.text
.endif

.type d1_panel_crc,%function
d1_panel_crc: /* x0 payload, x1 positive multiple of eight, w2 CRC state */
    add x1,x0,x1
.Lpanel_crc_next:
    ldr x3,[x0],#8
    crc32x w2,w2,x3
    cmp x0,x1
    b.lo .Lpanel_crc_next
    mov w0,w2
    ret
.size d1_panel_crc,.-d1_panel_crc

.ifdef D1_PARALLEL_OTHER_PANEL_CRC
.ifndef D1_PARALLEL_CHECKPOINT_CRC
.error "Other panel shard checks require the parallel checkpoint helper"
.endif
.type d1_read_other_panel_crc,%function
d1_read_other_panel_crc:
    ENTER
    ADDR x0,d1_other_crc_env
    bl getenv
    cbz x0,.Lother_crc_ready
    ldrb w1,[x0]
    cmp w1,#'1'
    cset w1,eq
    ADDR x0,d1_other_crc_enabled
    str w1,[x0]
.Lother_crc_ready:
    LEAVE
.size d1_read_other_panel_crc,.-d1_read_other_panel_crc
.section .rodata
d1_other_crc_env: .asciz "D1_PARALLEL_OTHER_PANEL_CRC"
d1_other_crc_stats: .asciz "d1_other_panel_crc_shards=%u\n"
.bss
.balign 4
d1_other_crc_enabled: .skip 4
d1_other_crc_shards: .skip 4
    .ifdef D1_ASYNC_BACKBONE_PANELS
/* The loader owns the Q6 count; the vision thread owns the other count. */
d1_q6_crc_shards: .skip 4
    .endif
.text
.endif

.ifdef D1_PARALLEL_Q4_PANEL_CRC
.ifndef D1_PARALLEL_CHECKPOINT_CRC
.error "Q4 shard checks require the parallel checkpoint helper"
.endif
.type d1_shard_crc,%function
d1_shard_crc: /* x0 payload, x1 bytes, w2 state, x3 page mask or zero */
    cbz x3,d1_panel_crc
    ENTER
    mov x19,x0
    mov x20,x1
    mov w21,w2
    mov x22,x3
.Lbounded_crc_chunk:
    mov x23,#(8*1024*1024)
    cmp x20,x23
    csel x23,x20,x23,lo
    mov x0,x19
    mov x1,x23
    mov w2,w21
    bl d1_panel_crc
    mov w21,w0
    mov x0,x19
    mov x1,x23
    mov x2,x22
    bl d1_panel_discard
    cbnz w0,file_error
    add x19,x19,x23
    subs x20,x20,x23
    b.ne .Lbounded_crc_chunk
    mov w0,w21
    LEAVE
.size d1_shard_crc,.-d1_shard_crc
.endif

.ifdef D1_PARALLEL_CHECKPOINT_CRC
/* The checked split is floor(file_bytes / 8192) * 4096. Each shard is
 * independently bound to the original checkpoint by its pinned CRC. */
.if MODEL_BYTES != 1099629888 || MODEL_CRC != 0xd53ce5f4 || D1_VISION_BYTES != 203031584 || D1_VISION_CRC != 0xfbef6164
.error "Checkpoint shard checks require the pinned D1 files"
.endif
.type d1_verify_checkpoint_shards,%function
d1_verify_checkpoint_shards: /* x0 file, x1 bytes, w2 first CRC, w3 second CRC */
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    mov x4,#0
.type d1_verify_bounded_shards,%function
d1_verify_bounded_shards: /* same arguments, x4 page mask for bounded residency */
    .endif
    ENTER
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    sub sp,sp,#48
    mov x25,x4
    str x4,[sp,#32]
    .else
    sub sp,sp,#32
    .endif
    mov x19,x0
    lsr x20,x1,#13
    lsl x20,x20,#12
    mov w22,w2
    mov w23,w3
    add x4,x19,x20
    sub x5,x1,x20
    stp x4,x5,[sp,#8]
    mov x0,sp
    mov x1,#0
    ADDR x2,d1_checkpoint_crc_worker
    add x3,sp,#8
    bl pthread_create
    mov w24,w0
    cbz w24,.Lshard_first
    add x0,sp,#8
    bl d1_checkpoint_crc_worker
.Lshard_first:
    mov x0,x19
    mov x1,x20
    mov w2,#-1
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    mov x3,x25
    bl d1_shard_crc
    .else
    bl d1_panel_crc
    .endif
    mvn w21,w0
    cbnz w24,.Lshard_verify
    ldr x0,[sp]
    mov x1,#0
    bl pthread_join
    cbnz w0,thread_error
.Lshard_verify:
    ldr w0,[sp,#24]
    cmp w21,w22
    b.ne checksum_error
    cmp w0,w23
    b.ne checksum_error
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    add sp,sp,#48
    .else
    add sp,sp,#32
    .endif
    LEAVE
.size d1_verify_checkpoint_shards,.-d1_verify_checkpoint_shards
.ifdef D1_PARALLEL_Q4_PANEL_CRC
.size d1_verify_bounded_shards,.-d1_verify_bounded_shards
.endif

.type d1_checkpoint_crc_worker,%function
d1_checkpoint_crc_worker:
    ENTER
    mov x19,x0
    ldp x0,x1,[x19]
    mov w2,#-1
    .ifdef D1_PARALLEL_Q4_PANEL_CRC
    ldr x3,[x19,#24]
    bl d1_shard_crc
    .else
    bl d1_panel_crc
    .endif
    mvn w0,w0
    str w0,[x19,#16]
    mov x0,#0
    LEAVE
.size d1_checkpoint_crc_worker,.-d1_checkpoint_crc_worker

.section .rodata
d1_parallel_crc_env: .asciz "D1_PARALLEL_CHECKPOINT_CRC"
.bss
.balign 4
d1_parallel_crc_enabled: .skip 4
.text
.endif
.ltorg

.section .rodata
d1_panel_header_error: .asciz "Invalid checkpoint-bound panel header.\n"
.text
