.ifdef D1_DEFER_MODEL_CRC
.ifndef D1_MODEL_EMBED_ROW_CRC
.error "Deferred model checks require the pinned embedding row verifier"
.endif

.section .rodata
.include "build/d1-deferred-crc/segments.inc"
d1_defer_env: .asciz "D1_DEFER_MODEL_CRC"
d1_defer_stats: .asciz "d1_deferred_model_crc tensors=%u bytes=%u\n"
.balign 8
d1_defer_model_spec:
    .word D1_DEFER_EAGER_COUNT,D1_DEFER_TENSOR_COUNT,D1_DEFER_EAGER_CRC,0
    .quad d1_defer_eager_table,d1_defer_tensor_table,d1_defer_verified,d1_defer_counts

.ifdef D1_DEFER_VISION_CRC
.include "build/d1-deferred-crc/projector.inc"
d1_defer_vis_env: .asciz "D1_DEFER_VISION_CRC"
d1_defer_vis_stats: .asciz "d1_deferred_vision_crc tensors=%u bytes=%u\n"
.balign 8
d1_defer_vis_spec:
    .word D1_DEFER_VIS_EAGER_COUNT,D1_DEFER_VIS_TENSOR_COUNT,D1_DEFER_VIS_EAGER_CRC,0
    .quad d1_defer_vis_eager_table,d1_defer_vis_tensor_table,d1_defer_vis_verified,d1_defer_vis_counts
.endif

.bss
.balign 8
d1_defer_enabled: .skip 4
d1_defer_counts: .skip 8
d1_defer_page_mask: .skip 8
d1_defer_verified: .skip D1_DEFER_TENSOR_COUNT
.ifdef D1_DEFER_VISION_CRC
.balign 8
d1_defer_vis_enabled: .skip 4
d1_defer_vis_counts: .skip 8
d1_defer_vis_verified: .skip D1_DEFER_VIS_TENSOR_COUNT
.endif

.text
.type d1_read_deferred_crc,%function
d1_read_deferred_crc:
    ENTER
    ADDR x0,d1_defer_env
    bl getenv
    cbz x0,.Ldefer_config_done
    ldrb w1,[x0]
    cmp w1,#'1'
    b.ne .Ldefer_config_done
    ADDR x0,d1_embed_crc_enabled
    ldr w1,[x0]
    cbz w1,d1_decision_invalid
    ADDR x0,d1_defer_enabled
    str w1,[x0]
.Ldefer_config_done:
    LEAVE
.size d1_read_deferred_crc,.-d1_read_deferred_crc

.ifdef D1_DEFER_VISION_CRC
.type d1_read_deferred_vision_crc,%function
d1_read_deferred_vision_crc:
    ENTER
    ADDR x0,d1_defer_vis_env
    bl getenv
    cbz x0,.Ldefer_vis_config_done
    ldrb w1,[x0]
    cmp w1,#'1'
    cset w1,eq
    ADDR x0,d1_defer_vis_enabled
    str w1,[x0]
.Ldefer_vis_config_done:
    LEAVE
.size d1_read_deferred_vision_crc,.-d1_read_deferred_vision_crc
.endif

/* x0 start, x1 end, x2 page mask, w3 protection. Do not alter pages
 * shared with neighboring tensors or GGUF metadata. */
.type d1_defer_protect,%function
d1_defer_protect:
    add x0,x0,x2
    bic x0,x0,x2
    bic x1,x1,x2
    subs x1,x1,x0
    b.ls .Ldefer_empty_pages
    mov w2,w3
    b mprotect
.Ldefer_empty_pages:
    mov w0,#0
    ret
.size d1_defer_protect,.-d1_defer_protect

.type d1_verify_eager_segments,%function
d1_verify_eager_segments: /* x0 original mapping */
    ADDR x1,d1_defer_model_spec
    .ifdef D1_DEFER_VISION_CRC
    b .Ldefer_eager_common
.type d1_verify_eager_vision_segments,%function
d1_verify_eager_vision_segments:
    ADDR x1,d1_defer_vis_spec
.Ldefer_eager_common:
    .endif
    ENTER
    mov x19,x0
    mov x23,x1
    ldr x20,[x23,#16]
    ldr w21,[x23]
    mov w22,#-1
.Ldefer_eager_crc:
    ldp w0,w1,[x20],#8
    add x0,x19,x0
    mov w2,w22
    bl d1_panel_crc
    mov w22,w0
    subs w21,w21,#1
    b.ne .Ldefer_eager_crc
    mvn w22,w22
    ldr w0,[x23,#8]
    cmp w22,w0
    b.ne checksum_error
    bl getpagesize
    sub x22,x0,#1
    ADDR x0,d1_defer_page_mask
    str x22,[x0]
    ldr x20,[x23,#24]
    ldr w21,[x23,#4]
.Ldefer_protect_next:
    ldp w0,w1,[x20],#16
    add x0,x19,x0
    add x1,x19,x1
    mov x2,x22
    mov w3,#0 /* PROT_NONE */
    bl d1_defer_protect
    cbnz w0,file_error
    subs w21,w21,#1
    b.ne .Ldefer_protect_next
    LEAVE
.size d1_verify_eager_segments,.-d1_verify_eager_segments
.ifdef D1_DEFER_VISION_CRC
.size d1_verify_eager_vision_segments,.-d1_verify_eager_vision_segments
.endif

/* Main thread only, before any raw kernel is called or workers see the job.
 * x0 may point inside a tensor when convolution projects a slice. */
.type d1_verify_raw_tensor,%function
d1_verify_raw_tensor:
    mov x7,x0
    ADDR x1,d1_defer_enabled
    ldr w1,[x1]
    cbz w1,.Ldefer_try_vision
    ADDR x1,model_base
    ldr x1,[x1]
    sub x0,x7,x1
    ldr x2,=MODEL_BYTES
    cmp x0,x2
    b.hs .Ldefer_try_vision
    ADDR x7,d1_defer_model_spec
    b .Ldefer_lookup
.Ldefer_try_vision:
    .ifdef D1_DEFER_VISION_CRC
    ADDR x1,d1_defer_vis_enabled
    ldr w1,[x1]
    cbz w1,.Ldefer_raw_done
    ADDR x1,d1_vis_base
    ldr x1,[x1]
    sub x0,x7,x1
    ldr x2,=D1_VISION_BYTES
    cmp x0,x2
    b.hs .Ldefer_raw_done
    ADDR x7,d1_defer_vis_spec
    .else
    ret
    .endif
.Ldefer_lookup:
    ldp x2,x5,[x7,#24]
    ldr w6,[x7,#4]
.Ldefer_find:
    ldp w3,w4,[x2]
    cmp x0,x3
    b.lo .Ldefer_raw_done
    cmp x0,x4
    b.lo .Ldefer_match
    add x2,x2,#16
    add x5,x5,#1
    subs w6,w6,#1
    b.ne .Ldefer_find
.Ldefer_raw_done:
    ret
.Ldefer_match:
    ldrb w0,[x5]
    cbnz w0,.Ldefer_raw_done
    ENTER
    add x19,x1,x3
    add x20,x1,x4
    mov x21,x5
    ldr x23,[x7,#40]
    ldr w22,[x2,#8]
    mov x0,x19
    mov x1,x20
    ADDR x2,d1_defer_page_mask
    ldr x2,[x2]
    mov w3,#1 /* PROT_READ */
    bl d1_defer_protect
    cbnz w0,file_error
    sub x20,x20,x19
    mov x0,x19
    mov x1,x20
    mov w2,#-1
    bl d1_panel_crc
    mvn w0,w0
    cmp w0,w22
    b.ne checksum_error
    mov w0,#1
    strb w0,[x21]
    ldp w1,w2,[x23]
    add w1,w1,#1
    add w2,w2,w20
    stp w1,w2,[x23]
    LEAVE
.size d1_verify_raw_tensor,.-d1_verify_raw_tensor
.ltorg
.endif
