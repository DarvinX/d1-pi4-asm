.ifdef D1_MODEL_EMBED_ROW_CRC
.ifndef D1_PARALLEL_CHECKPOINT_CRC
.error "Embedding row verification requires the parallel checkpoint helper"
.endif
.include "build/d1-embed-crc/rows.inc"
.if VOCAB != D1_EMBED_ROW_COUNT || DIM != 2048 || D1_EMBED_ROW_BYTES != 1680
.error "Embedding row CRC table requires the pinned tied Q6_K tensor"
.endif

.text
.type d1_read_embedding_crc,%function
d1_read_embedding_crc:
    ENTER
    ADDR x0,d1_embed_crc_env
    bl getenv
    cbz x0,.Lembed_crc_config_done
    ldrb w1,[x0]
    cmp w1,#'1'
    b.ne .Lembed_crc_config_done
    ADDR x0,d1_parallel_crc_env
    bl getenv
    cbz x0,d1_decision_invalid
    ldrb w1,[x0]
    cmp w1,#'1'
    b.ne d1_decision_invalid
    ldrb w1,[x0,#1]
    cbnz w1,d1_decision_invalid
    ADDR x0,d1_embed_crc_enabled
    mov w1,#1
    str w1,[x0]
.Lembed_crc_config_done:
    LEAVE
.size d1_read_embedding_crc,.-d1_read_embedding_crc

.type d1_verify_model_segments,%function
d1_verify_model_segments: /* x0 original mapping, verify every byte outside embedding */
    .ifdef D1_DEFER_MODEL_CRC
    ADDR x1,d1_defer_enabled
    ldr w1,[x1]
    cbnz w1,d1_verify_eager_segments
    .endif
    ENTER
    mov x19,x0
    ldr x1,=D1_EMBED_ROW_FIRST
    mov w2,#-1
    bl d1_panel_crc
    mvn w0,w0
    ldr w1,=D1_EMBED_PREFIX_CRC
    cmp w0,w1
    b.ne checksum_error
    ldr x0,=D1_EMBED_ROW_END
    add x0,x19,x0
    ldr x1,=MODEL_BYTES-D1_EMBED_ROW_END
    ldr w2,=D1_EMBED_TAIL_LEFT_CRC
    ldr w3,=D1_EMBED_TAIL_RIGHT_CRC
    bl d1_verify_checkpoint_shards
    LEAVE
.size d1_verify_model_segments,.-d1_verify_model_segments

.type d1_verify_embedding_row,%function
d1_verify_embedding_row: /* x0 row, w1 token ID -> x0 original row */
    ADDR x2,d1_embed_crc_enabled
    ldr w2,[x2]
    cbz w2,.Lembed_row_return
    ENTER
    mov x19,x0
    ldr w2,=D1_EMBED_ROW_COUNT
    cmp w1,w2
    b.hs checksum_error
    ADDR x2,d1_embed_crc_table
    ldr w20,[x2,x1,lsl #2]
    mov x1,#D1_EMBED_ROW_BYTES
    mov w2,#-1
    bl d1_panel_crc
    mvn w0,w0
    cmp w0,w20
    b.ne checksum_error
    ADDR x1,d1_embed_crc_rows
    ldr w0,[x1]
    add w0,w0,#1
    str w0,[x1]
    mov x0,x19
    LEAVE
.Lembed_row_return:
    ret
.size d1_verify_embedding_row,.-d1_verify_embedding_row

.type d1_verify_label_rows,%function
d1_verify_label_rows: /* x0 tied table, x1 IDs, w2 count -> x0 tied table */
    ADDR x3,d1_embed_crc_enabled
    ldr w3,[x3]
    cbz w3,.Lverify_labels_return
    ENTER
    mov x19,x0
    mov x20,x1
    mov w21,w2
    mov x22,#D1_EMBED_ROW_BYTES
.Lverify_label_row:
    ldr w1,[x20],#4
    madd x0,x1,x22,x19
    bl d1_verify_embedding_row
    subs w21,w21,#1
    b.ne .Lverify_label_row
    mov x0,x19
    LEAVE
.Lverify_labels_return:
    ret
.size d1_verify_label_rows,.-d1_verify_label_rows
.ltorg

.section .rodata
d1_embed_crc_env: .asciz "D1_MODEL_EMBED_ROW_CRC"
d1_embed_crc_stats: .asciz "d1_model_embed_crc_rows=%u\n"
.balign 64
d1_embed_crc_table:
    .incbin "build/d1-embed-crc/rows.bin"
    .if .-d1_embed_crc_table != 4*D1_EMBED_ROW_COUNT
    .error "Invalid embedding CRC table length"
    .endif
.bss
.balign 4
d1_embed_crc_enabled: .skip 4
d1_embed_crc_rows: .skip 4
.text
.endif
