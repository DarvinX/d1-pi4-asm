.text
.macro ROUTE_QUERY_COPY row,slot,scatter=0
    .if \scatter
    add x0,x25,#\row,lsl #12
    add x1,x22,#\slot*256
    .else
    add x0,x21,#\slot*256
    add x1,x25,#\row,lsl #12
    .endif
    mov x2,#256
    bl memcpy
.endm

.type d1_vision_attn_route_job,%function
d1_vision_attn_route_job: /* One claimant prepares a KV head and its two query heads. */
    ENTER
    sub sp,sp,#189,lsl #12
    ADDR x19,d1_vis_attn_head
    add x21,sp,#64 /* Q, four rows */
    add x22,sp,#1088 /* output, four rows */
    add x0,sp,#4096 /* quad-major K, V, mean K, mean V, radius */
    stp x21,x0,[sp]
    add x0,x0,#75,lsl #12
    add x1,x0,#75,lsl #12
    stp x0,x1,[sp,#16]
    add x0,x1,#18,lsl #12
    add x0,x0,#3072
    add x1,x0,#18,lsl #12
    add x1,x1,#3072
    stp x0,x1,[sp,#32]
    str x22,[sp,#48]
    ADDR x0,d1_vis_route_budget
    ldr w0,[x0]
    str w0,[sp,#56]
.Lroute_claim:
    ldaxr w23,[x19]
    add w0,w23,#1
    stlxr w1,w0,[x19]
    cbnz w1,.Lroute_claim
    cmp w23,#8
    b.hs .Lroute_job_done
    ADDR x0,d1_vis_norm
    mov w1,#1200
    lsl x1,x1,#8
    madd x0,x23,x1,x0
    add x1,x0,#600,lsl #12
    mov x2,sp
    bl d1_route_pack_head
    ADDR x24,d1_vis_work
    add x24,x24,x23,lsl #9
    mov w26,#2
.Lroute_job_head:
    mov x25,x24
    mov w28,#15
.Lroute_job_y:
    mov w27,#20
.Lroute_job_x:
    ROUTE_QUERY_COPY 0,0
    ROUTE_QUERY_COPY 1,1
    ROUTE_QUERY_COPY 40,2
    ROUTE_QUERY_COPY 41,3
    mov x0,sp
    bl d1_route_attention4
    ROUTE_QUERY_COPY 0,0,1
    ROUTE_QUERY_COPY 1,1,1
    ROUTE_QUERY_COPY 40,2,1
    ROUTE_QUERY_COPY 41,3,1
    add x25,x25,#2,lsl #12
    subs w27,w27,#1
    b.ne .Lroute_job_x
    add x25,x25,#40,lsl #12
    subs w28,w28,#1
    b.ne .Lroute_job_y
    add x24,x24,#256
    subs w26,w26,#1
    b.ne .Lroute_job_head
    b .Lroute_claim
.Lroute_job_done:
    add sp,sp,#189,lsl #12
    LEAVE
.size d1_vision_attn_route_job,.-d1_vision_attn_route_job

.type d1_vis_route_report,%function
d1_vis_route_report:
    ADDR x0,d1_vis_route_layers
    ldr w2,[x0]
    cbz w2,.Lroute_report_done
    mov w0,#2
    ADDR x1,d1_vis_route_stats
    b dprintf
.Lroute_report_done:
    ret
.size d1_vis_route_report,.-d1_vis_route_report

.section .rodata
d1_vis_route_env: .asciz "D1_VISION_ROUTE"
d1_vis_route_stats: .asciz "d1_vision_routed layers=%u\n"
.bss
.balign 4
d1_vis_route_enabled: .skip 4
d1_vis_route_budget: .skip 4
d1_vis_route_layers: .skip 4
.text
