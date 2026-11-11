/* Offline Q6_K panels, bound to the verified original model. */
.equ D1_Q6_I16_LAYERS,11
.equ D1_Q6_I16_LAYER_BYTES,(2048/8)*(7680/256)*D1_Q6_I16_TILE_BYTES
.equ D1_Q6_I16_FILE_BYTES,4096+D1_Q6_I16_LAYERS*D1_Q6_I16_LAYER_BYTES

.text
.type d1_q6_i16_map,%function
d1_q6_i16_map:
    ENTER
    ADDR x0,d1_q6_i16_env
    bl getenv
    cbz x0,.Lq6_map_done
    ldrb w1,[x0]
    cbz w1,.Lq6_map_done
    ldr x1,=D1_Q6_I16_FILE_BYTES
    ADDR x2,d1_q6_i16_magic
    bl d1_panel_map_file
    mov x20,x0
    .ifdef D1_SHORT_PACKED_ROWS
    ADDR x0,d1_q6_i16_min_env
    bl getenv
    mov w19,#48
    cbz x0,.Lq6_map_threshold
    mov x21,x0
    ADDR x1,d1_q6_i16_min28
    bl strcmp
    cbnz w0,.Lq6_map_test7
    mov w19,#28
    b .Lq6_map_threshold
.Lq6_map_test7:
    mov x0,x21
    ADDR x1,d1_q6_i16_min7
    bl strcmp
    cbnz w0,.Lq6_map_test48
    mov w19,#7
    b .Lq6_map_threshold
.Lq6_map_test48:
    mov x0,x21
    ADDR x1,d1_q6_i16_min48
    bl strcmp
    cbnz w0,.Lq6_invalid_threshold
.Lq6_map_threshold:
    ADDR x0,d1_q6_i16_min
    str w19,[x0]
    .endif

    /* The original GGUF CRC is already verified. Release only pages
     * wholly inside a replaced tensor. Fallback can fault them back in. */
    bl getpagesize
    sub x23,x0,#1
    ADDR x19,model_base
    ldr x19,[x19]
    ADDR x21,layer_tensors
    mov w22,#LAYERS-1
    ADDR x24,d1_q6_i16_index
    add x25,x20,#4096
    mov w26,#0
    ldr x27,=D1_Q6_I16_LAYER_BYTES
.Lq6_index_layer:
    ldr w0,[x21,#6*32+16]
    cmp w0,#14
    b.ne .Lq6_index_next
    ldp w0,w1,[x21,#6*32+8]
    mov w2,#7680
    cmp w0,w2
    b.ne .Lq6_invalid
    cmp w1,#2048
    b.ne .Lq6_invalid
    cmp w26,#D1_Q6_I16_LAYERS
    b.hs .Lq6_invalid
    ldr x0,[x21,#6*32]
    ldr x1,[x21,#6*32+24]
    add x0,x19,x0
    stp x0,x25,[x24],#16
    mov x2,x23
    bl d1_panel_discard
    cbnz w0,file_error
    add x25,x25,x27
    add w26,w26,#1
.Lq6_index_next:
    add x21,x21,#576
    subs w22,w22,#1
    b.ne .Lq6_index_layer
    cmp w26,#D1_Q6_I16_LAYERS
    b.ne .Lq6_invalid

    .ifdef D1_PARALLEL_OTHER_PANEL_CRC
    .if D1_Q6_I16_FILE_BYTES != 347385856
    .error "Q6 shard checks require the pinned Q6 sidecar"
    .endif
    ADDR x0,d1_other_crc_enabled
    ldr w0,[x0]
    cbz w0,.Lq6_serial_crc
    ldr w0,[x20,#12]
    ldr w1,=0x1f154b9a
    cmp w0,w1
    b.ne .Lq6_invalid
    add x0,x20,#4096
    sub x1,x25,x0
    ldr w2,=0x8453544d
    ldr w3,=0x1ef01892
    bl d1_verify_checkpoint_shards
    .ifdef D1_ASYNC_BACKBONE_PANELS
    ADDR x0,d1_q6_crc_shards
    .else
    ADDR x0,d1_other_crc_shards
    .endif
    mov w1,#2
    str w1,[x0]
    b .Lq6_crc_verified
.Lq6_serial_crc:
    .endif
    add x0,x20,#4096
    sub x1,x25,x0
    mov w2,#-1
    bl d1_panel_crc
    mvn w0,w0
    ldr w1,[x20,#12]
    cmp w0,w1
    b.ne .Lq6_invalid
    .ifdef D1_PARALLEL_OTHER_PANEL_CRC
.Lq6_crc_verified:
    .endif
    ADDR x0,d1_q6_i16_enabled
    mov w1,#1
    str w1,[x0]
.Lq6_map_done:
    LEAVE
.Lq6_invalid:
    ADDR x0,d1_q6_i16_error
    b fatal
    .ifdef D1_SHORT_PACKED_ROWS
.Lq6_invalid_threshold:
    ADDR x0,d1_q6_i16_min_error
    b fatal
    .endif
.size d1_q6_i16_map,.-d1_q6_i16_map

.type d1_q6_i16_find,%function
d1_q6_i16_find: /* original tensor address -> compact layer, or NULL */
    ADDR x1,d1_q6_i16_index
    mov w2,#D1_Q6_I16_LAYERS
.Lq6_find_layer:
    ldp x3,x4,[x1],#16
    cmp x0,x3
    b.eq .Lq6_found_layer
    subs w2,w2,#1
    b.ne .Lq6_find_layer
    mov x4,#0
.Lq6_found_layer:
    mov x0,x4
    ret
.size d1_q6_i16_find,.-d1_q6_i16_find

.type d1_q6_i16_pack_activations,%function
d1_q6_i16_pack_activations: /* x0 packed Q8_K, x1 panels, w2 rows, w3 row bytes */
    ENTER
    mov x19,x0
    mov x20,x1
    mov w21,w2
    mov w22,w3
    mov w23,#7680/D1_Q6_I16_KC
.Lq6_act_k:
    mov x24,x19
    mov w25,w21
.Lq6_act_tile:
    mov w26,#0
.Lq6_act_four:
    lsl x0,x26,#2
    madd x0,x0,x22,x24
    add x1,x20,x26,lsl #4
    mov x2,x22
    add x3,x20,#48
    add x3,x3,x26,lsl #11
    bl d1_pack_q8k_i16_4
    add w26,w26,#1
    cmp w26,#3
    b.ne .Lq6_act_four
    mov w0,#D1_Q6_I16_ACT_BYTES
    add x20,x20,x0
    add x24,x24,x22,lsl #3
    add x24,x24,x22,lsl #2
    subs w25,w25,#12
    b.ne .Lq6_act_tile
    add x19,x19,#320
    subs w23,w23,#1
    b.ne .Lq6_act_k
    LEAVE
.size d1_q6_i16_pack_activations,.-d1_q6_i16_pack_activations

.type d1_q6_i16_report,%function
d1_q6_i16_report:
    ADDR x0,d1_q6_i16_enabled
    ldr w0,[x0]
    cbz w0,.Lq6_report_done
    mov w0,#2
    ADDR x1,d1_q6_i16_stats
    ADDR x2,d1_q6_i16_batches
    ldr w2,[x2]
    ADDR x3,d1_q6_i16_fallback
    ldp w3,w4,[x3]
    b dprintf
.Lq6_report_done:
    ret
.size d1_q6_i16_report,.-d1_q6_i16_report
.ltorg

.section .rodata
.balign 8
d1_q6_i16_magic: .ascii "D1Q6I16\001"
d1_q6_i16_env: .asciz "D1_Q6_I16_PANELS"
    .ifdef D1_SHORT_PACKED_ROWS
d1_q6_i16_min_env: .asciz "D1_Q6_I16_MIN_ROWS"
d1_q6_i16_min7: .asciz "7"
d1_q6_i16_min28: .asciz "28"
d1_q6_i16_min48: .asciz "48"
d1_q6_i16_min_error: .asciz "Invalid Q6_K panel row threshold. Expected 7, 28 or 48.\n"
    .endif
d1_q6_i16_error: .asciz "Invalid checkpoint-bound Q6_K panel file.\n"
d1_q6_i16_stats: .asciz "d1_q6_i16 batches=%u fallback=%u fallback_rows=%u\n"
.bss
.balign 8
d1_q6_i16_index: .skip D1_Q6_I16_LAYERS*16
d1_q6_i16_for_job: .skip 8
d1_q6_i16_enabled: .skip 4
d1_q6_i16_batches: .skip 4
d1_q6_i16_fallback: .skip 8
    .ifdef D1_SHORT_PACKED_ROWS
d1_q6_i16_min: .skip 4
    .endif
.text
