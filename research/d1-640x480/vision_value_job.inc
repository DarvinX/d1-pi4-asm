/* The dense shape has no query or key tails. The existing scorer still
 * produces four rows at a time. Each value tile consumes three such groups. */
.text
.type d1_vision_dispatch_attention,%function
d1_vision_dispatch_attention:
    ADDR x0,d1_vis_route_budget
    ldr w0,[x0]
    cbz w0,.Lattention_dense
    ADDR x0,d1_vis_attn4_enabled
    ldr w0,[x0]
    cbz w0,.Lattention_dense
    ADDR x0,d1_vis_keys_this_layer
    ldr w0,[x0]
    cmp w0,#1200
    b.ne .Lattention_dense
    VIS_ROWS w0
    cmp w0,#1200
    b.ne .Lattention_dense
    ADDR x0,d1_vis_pool_this_layer
    ldr w0,[x0]
    cbnz w0,.Lattention_dense
    ADDR x0,d1_vis_route_layers
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    mov w0,#5
    b d1_batch_dispatch
.Lattention_dense:
    ADDR x0,d1_vis_value12_enabled
    ldr w0,[x0]
    cbz w0,.Lattention_job4
    ADDR x0,d1_vis_attn4_enabled
    ldr w0,[x0]
    cbz w0,.Lattention_job4
    ADDR x0,d1_vis_pool_this_layer
    ldr w0,[x0]
    .ifdef D1_VISION_POOLED12
    cbz w0,.Lattention_unpooled12
    ADDR x0,d1_vis_pooled12_enabled
    ldr w0,[x0]
    cbnz w0,.Lattention_tiled_ready
    b .Lattention_job4
.Lattention_unpooled12:
    .endif
    ADDR x1,d1_vis_score_packed_enabled
    ldr w1,[x1]
    orr w0,w0,w1
    cbnz w0,.Lattention_job4
    ADDR x0,d1_vis_keys_this_layer
    ldr w0,[x0]
    cmp w0,#1200
    b.ne .Lattention_job4
    VIS_ROWS w0
    cmp w0,#1200
    .ifdef D1_VISION_QUERY_GROUP
    b.eq .Lattention_dense_rows
    ADDR x0,d1_vis_query_active
    ldr w0,[x0]
    cbz w0,.Lattention_job4
.Lattention_dense_rows:
    .else
    b.ne .Lattention_job4
    .endif
    .ifdef D1_VISION_POOLED12
.Lattention_tiled_ready:
    .endif
    ADDR x0,d1_vis_value12_layers
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    ADDR x0,d1_vis_score12_enabled
    ldr w1,[x0]
    ADDR x0,d1_vis_score12_layers
    ldr w2,[x0]
    add w2,w2,w1
    str w2,[x0]
    mov w0,#4
    b d1_batch_dispatch
.Lattention_job4:
    mov w0,#2
    b d1_batch_dispatch
.size d1_vision_dispatch_attention,.-d1_vision_dispatch_attention

.type d1_vision_attn_job12,%function
d1_vision_attn_job12:
    ENTER
    /* Score, packed V, and value scratch occupy the first 104 pages.
     * Optional score12 adds 75 pages for packed K and one for packed Q. */
    sub sp,sp,#180,lsl #12
    ADDR x19,d1_vis_attn_head
    add x23,sp,#14,lsl #12
    add x23,x23,#256
    add x27,x23,#75,lsl #12
.Lvalue12_claim:
    ldaxr w20,[x19]
    add w0,w20,#1
    stlxr w1,w0,[x19]
    cbnz w1,.Lvalue12_claim
    cmp w20,#8
    b.hs .Lvalue12_done
    ADDR x22,d1_vis_norm
    .ifdef D1_VISION_POOLED12
    ADDR x0,d1_vis_keys_this_layer
    ldr w2,[x0]
    lsl x0,x2,#8
    .else
    mov w0,#1200
    lsl x0,x0,#8
    .endif
    madd x22,x20,x0,x22
    add x0,x22,#600,lsl #12
    mov x1,x23
    .ifndef D1_VISION_POOLED12
    mov w2,#1200
    .endif
    bl d1_pack_value_head
    ADDR x0,d1_vis_score12_enabled
    ldr w0,[x0]
    cbz w0,.Lvalue12_keys_ready
    mov x0,x22
    add x1,sp,#104,lsl #12
    .ifdef D1_VISION_POOLED12
    ADDR x2,d1_vis_keys_this_layer
    ldr w2,[x2]
    bl d1_pack_score_keys8_counted
    .else
    bl d1_pack_score_keys8
    .endif
.Lvalue12_keys_ready:
    ADDR x25,d1_vis_work
    add x25,x25,x20,lsl #9
    mov w26,#2
.Lvalue12_head:
    mov x21,x25
    .ifdef D1_VISION_QUERY_GROUP
    VIS_ROWS w24
    mov w0,#12
    udiv w24,w24,w0
    .else
    mov w24,#100
    .endif
.Lvalue12_query:
    ADDR x0,d1_vis_score12_enabled
    ldr w0,[x0]
    cbz w0,.Lvalue12_four_scores
    mov x0,x21
    add x1,sp,#104,lsl #12
    mov x2,sp
    add x3,sp,#179,lsl #12
    mov x4,#4096
    .ifdef D1_VISION_POOLED12
    ADDR x5,d1_vis_pool_this_layer
    ldr w5,[x5]
    cbz w5,.Lvalue12_dense_score
    ADDR x5,d1_vis_keys_this_layer
    ldr w5,[x5]
    ADDR x6,d1_vis_pool_bias
    bl d1_attention_score12_pooled
    b .Lvalue12_accumulate
.Lvalue12_dense_score:
    .endif
    bl d1_attention_score12
    b .Lvalue12_accumulate
.Lvalue12_four_scores:
    mov w28,#0
.Lvalue12_score:
    add x0,x21,x28,lsl #14
    mov x1,x22
    mov x4,#19200
    mul x4,x28,x4
    add x4,sp,x4
    mov w5,#1200
    mov x6,#4096
    mov x7,#256
    bl d1_vision_attention4_probs
    add w28,w28,#1
    cmp w28,#3
    b.ne .Lvalue12_score
.Lvalue12_accumulate:
    mov x0,sp
    mov x1,x23
    mov x2,x21
    .ifdef D1_VISION_POOLED12
    ADDR x3,d1_vis_keys_this_layer
    ldr w3,[x3]
    .else
    mov w3,#1200
    .endif
    mov x4,#4096
    mov x5,x27
    mov x6,#4800
    bl d1_vision_value12
    add x21,x21,#12,lsl #12
    subs w24,w24,#1
    b.ne .Lvalue12_query
    add x25,x25,#256
    subs w26,w26,#1
    b.ne .Lvalue12_head
    b .Lvalue12_claim
.Lvalue12_done:
    add sp,sp,#180,lsl #12
    LEAVE
.size d1_vision_attn_job12,.-d1_vision_attn_job12

.type d1_report_value12,%function
d1_report_value12:
    ADDR x0,d1_vis_value12_enabled
    ldr w0,[x0]
    cbz w0,.Lvalue12_unreported
    mov w0,#2
    ADDR x1,d1_vis_value12_count
    ADDR x2,d1_vis_value12_layers
    ldr w2,[x2]
    b dprintf
.Lvalue12_unreported:
    ret
.size d1_report_value12,.-d1_report_value12

.type d1_vis_score12_report,%function
d1_vis_score12_report:
    ADDR x0,d1_vis_score12_layers
    ldr w2,[x0]
    cbz w2,.Lscore12_unreported
    mov w0,#2
    ADDR x1,d1_vis_score12_count
    b dprintf
.Lscore12_unreported:
    ret
.size d1_vis_score12_report,.-d1_vis_score12_report

.section .rodata
d1_vis_value12_env: .asciz "D1_VISION_VALUE12"
d1_vis_value12_count: .asciz "d1_vision_value12 layers=%u\n"
d1_vis_score12_env: .asciz "D1_VISION_SCORE12"
d1_vis_score12_count: .asciz "d1_vision_score12 layers=%u\n"
.bss
.balign 4
d1_vis_value12_enabled: .skip 4
d1_vis_value12_layers: .skip 4
d1_vis_score12_enabled: .skip 4
d1_vis_score12_layers: .skip 4
.text
