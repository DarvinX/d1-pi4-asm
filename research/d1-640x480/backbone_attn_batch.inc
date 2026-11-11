/* D1 attention keeps the original score and value reductions. It
 * batches the projections and assigns independent query rows to workers. */
.text
d1_read_attn_flag:
    ENTER
    ADDR x0,d1_attn_env
    bl getenv
    cbz x0,1f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_attn_enabled
    str w0,[x1]
1:  LEAVE

d1_read_tail_flag:
    ENTER
    ADDR x0,d1_ffn_enabled
    ldr w1,[x0]
    cbz w1,1f
    ADDR x0,d1_conv_enabled
    ldr w1,[x0]
    cbz w1,1f
    ADDR x0,d1_attn_enabled
    ldr w1,[x0]
    cbz w1,1f
    ADDR x0,d1_tail_env
    bl getenv
    cbz x0,1f
    ldrb w0,[x0]
    cmp w0,#'1'
    cset w0,eq
    ADDR x1,d1_tail_enabled
    str w0,[x1]
1:  LEAVE

.macro D1_ROPE_COPY src,dst,offset
    ADDR x0,\dst
    add x0,x0,x20,lsl #9
    .if \offset
    add x0,x0,#256
    .endif
    ADDR x1,\src
    mov x2,#256
    bl memcpy
.endm

.type d1_prepare_rope,%function
d1_prepare_rope: /* same coefficients, once for all attention layers */
    ENTER
    mov w19,w0
    mov w20,#0
.Lrope_cache_pos:
    mov w0,w20
    bl make_rope
    D1_ROPE_COPY cos_global,d1_rope_global,0
    D1_ROPE_COPY sin_global,d1_rope_global,1
    D1_ROPE_COPY cos_window,d1_rope_window,0
    D1_ROPE_COPY sin_window,d1_rope_window,1
    add w20,w20,#1
    cmp w20,w19
    b.lo .Lrope_cache_pos
    LEAVE
.size d1_prepare_rope,.-d1_prepare_rope

.macro D1_ATTN_MAT slot,input,output,suffix=0
    add x0,x19,#(\slot)*32
    ADDR x1,\input
    ADDR x2,\output
    mov w3,w20
    .if \suffix
    add x1,x1,x28,lsl #13
    .if \slot == 16
    add x2,x2,x28,lsl #6
    .else
    add x2,x2,x28,lsl #13
    .endif
    sub w3,w3,w28
    .endif
    bl d1_matrix_batch
.endm

.ifdef D1_SPARSE_WORK
.macro D1_ATTN_INDEXED slot,input,output
    add x0,x19,#(\slot)*32
    ADDR x1,\input
    ADDR x2,\output
    ADDR x3,d1_sparse_active
    ADDR x4,d1_sparse_sizes
    ldr w4,[x4]
    mov x5,#0
    mov x6,#0
    bl d1_matrix_batch_indexed
.endm

.macro D1_ATTN_SHARED slot,output
    add x0,x19,#(\slot)*32
    ADDR x1,d1_batch_norm
    ADDR x2,\output
    ADDR x3,d1_frozen_kv_list
    ADDR x4,d1_frozen_ops_sizes
    ldr w4,[x4]
    mov x5,#0
    mov x6,#0
    bl d1_matrix_batch_indexed
    ADDR x0,\output
    ADDR x1,d1_frozen_kv_owner
    ADDR x2,d1_image_first
    ldr w2,[x2]
    mov w3,#KW*4
    mov w4,#KW*4
    bl d1_frozen_projection_scatter
.endm

.macro D1_ATTN_SKIP_FROZEN position,target
    ADDR x0,d1_sparse_live
    ldr w0,[x0]
    cbz w0,.Lattn_row_live\@
    ADDR x0,d1_sparse_mask
    ldrb w0,[x0,\position]
    cbz w0,\target
.Lattn_row_live\@:
.endm
.endif

.type d1_attention_batch,%function
d1_attention_batch: /* descriptor, positions, type, first live query */
    ENTER
    mov x19,x0
    mov w20,w1
    mov w21,w2
    mov w28,w3
    mov w22,#0
.Lba_norm_pos:
    mov x0,x19
    ADDR x1,d1_hidden
    add x1,x1,x22,lsl #13
    ADDR x2,d1_batch_norm
    add x2,x2,x22,lsl #13
    mov w3,#DIM
    bl norm
    add w22,w22,#1
    cmp w22,w20
    b.lo .Lba_norm_pos

    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_sparse_live
    ldr w0,[x0]
    cbz w0,.Lba_dense_q
    D1_ATTN_INDEXED 10,d1_batch_norm,d1_queries
    b .Lba_project_kv
.Lba_dense_q:
    .endif
    D1_ATTN_MAT 10,d1_batch_norm,d1_queries,1
.Lba_project_kv:
    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_sparse_live
    ldr w0,[x0]
    cbz w0,.Lba_dense_kv
    ADDR x0,d1_frozen_ops_enabled
    ldr w0,[x0]
    cbz w0,.Lba_dense_kv
    D1_ATTN_SHARED 11,d1_keys
    D1_ATTN_SHARED 12,d1_values
    b .Lba_project_gate
.Lba_dense_kv:
    .endif
    D1_ATTN_MAT 11,d1_batch_norm,d1_keys
    D1_ATTN_MAT 12,d1_batch_norm,d1_values
    .ifdef D1_SPARSE_WORK
.Lba_project_gate:
    ADDR x0,d1_sparse_live
    ldr w0,[x0]
    cbz w0,.Lba_dense_gate
    D1_ATTN_INDEXED 16,d1_batch_norm,d1_gates
    b .Lba_qk_prepare
.Lba_dense_gate:
    .endif
    D1_ATTN_MAT 16,d1_batch_norm,d1_gates,1

.Lba_qk_prepare:
    mov w22,#0
.Lba_qk_pos:
    ADDR x23,d1_queries
    ADDR x24,d1_keys
    add x23,x23,x22,lsl #13
    add x24,x24,x22,lsl #11
    mov w25,#0
    cmp w22,w28
    b.lo .Lba_key_head
    .ifdef D1_SPARSE_WORK
    D1_ATTN_SKIP_FROZEN x22,.Lba_key_head
    .endif
.Lba_query_head:
    add x0,x19,#14*32
    add x1,x23,x25,lsl #9
    mov x2,x1
    mov w3,#HD
    bl norm
    add w25,w25,#1
    cmp w25,#HEADS
    b.lo .Lba_query_head
    mov w25,#0
.Lba_key_head:
    add x0,x19,#15*32
    add x1,x24,x25,lsl #9
    mov x2,x1
    mov w3,#HD
    bl norm
    add w25,w25,#1
    cmp w25,#KV
    b.lo .Lba_key_head
    ADDR x26,d1_rope_global
    cmp w21,#2
    b.ne .Lba_rotate
    ADDR x26,d1_rope_window
.Lba_rotate:
    add x26,x26,x22,lsl #9
    add x27,x26,#256
    cmp w22,w28
    b.lo .Lba_rotate_key
    .ifdef D1_SPARSE_WORK
    D1_ATTN_SKIP_FROZEN x22,.Lba_rotate_key
    .endif
    mov x0,x23
    mov x1,x26
    mov x2,x27
    mov w3,#HEADS
    bl rotate
.Lba_rotate_key:
    mov x0,x24
    mov x1,x26
    mov x2,x27
    mov w3,#KV
    bl rotate
    add w22,w22,#1
    cmp w22,w20
    b.lo .Lba_qk_pos

    ADDR x0,d1_attn_job
    stp w21,w20,[x0]
    str w28,[x0,#8]
    mov w0,#3
    bl d1_batch_dispatch
    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_sparse_live
    ldr w0,[x0]
    cbz w0,.Lba_dense_out
    D1_ATTN_INDEXED 13,d1_batch_norm,d1_batch_gate
    LEAVE
.Lba_dense_out:
    .endif
    D1_ATTN_MAT 13,d1_batch_norm,d1_batch_gate,1
    LEAVE
.size d1_attention_batch,.-d1_attention_batch

.type d1_attention_worker,%function
d1_attention_worker:
    ENTER
    ADDR x19,d1_attn_job
    ldp w20,w21,[x19]
    add x19,x19,#8
    ldr x24,=((CTX+3)&~3)*4
    sub sp,sp,x24
.Lba_claim:
    ldaxr w22,[x19]
    add w23,w22,#2
    stlxr w0,w23,[x19]
    cbnz w0,.Lba_claim
    cmp w22,w21
    b.hs .Lba_finished
    cmp w23,w21
    csel w23,w23,w21,lo
.Lba_query:
    .ifdef D1_SPARSE_WORK
    D1_ATTN_SKIP_FROZEN x22,.Lba_query_done
    .endif
    mov w0,w22
    mov w1,w20
    mov w2,w21
    ADDR x3,d1_batch_norm
    add x3,x3,x22,lsl #13
    mov x4,sp
    bl d1_attention_row
.Lba_query_done:
    add w22,w22,#1
    cmp w22,w23
    b.lo .Lba_query
    b .Lba_claim
.Lba_finished:
    add sp,sp,x24
    LEAVE
.size d1_attention_worker,.-d1_attention_worker
.ltorg

.section .rodata
d1_attn_env: .asciz "D1_ATTN_BATCH"
d1_tail_env: .asciz "D1_TAIL_PRUNE"
.bss
.balign 64
d1_attn_job: .skip 64
d1_attn_enabled: .skip 4
d1_tail_enabled: .skip 4
.balign 64
d1_rope_global: .skip CTX*512
d1_rope_window: .skip CTX*512
.text
