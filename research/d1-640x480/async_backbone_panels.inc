.ifdef D1_ASYNC_BACKBONE_PANELS
.text
.type d1_begin_backbone_panels,%function
d1_begin_backbone_panels:
    ENTER
    ADDR x0,d1_async_loader_env
    bl getenv
    cbz x0,.Lload_backbone_sync
    ldrb w1,[x0]
    cmp w1,#'1'
    b.ne .Lload_backbone_sync
    ADDR x0,d1_async_loader_thread
    mov x1,#0
    ADDR x2,d1_backbone_panel_loader
    mov x3,#0
    bl pthread_create
    cbnz w0,.Lload_backbone_sync
    ADDR x0,d1_async_loader_active
    mov w1,#1
    str w1,[x0]
    LEAVE
.Lload_backbone_sync:
    bl d1_q6_i16_map
    bl d1_q4_i8_map
    LEAVE
.size d1_begin_backbone_panels,.-d1_begin_backbone_panels

.type d1_backbone_panel_loader,%function
d1_backbone_panel_loader:
    ENTER
    /* On Linux this lowers only the loader's thread priority. */
    mov w0,#0 /* PRIO_PROCESS */
    mov w1,#0
    mov w2,#10
    bl setpriority
    mov w0,#0
    mov w1,#0
    bl getpriority
    ADDR x1,d1_async_loader_nice
    str w0,[x1]
    bl d1_q6_i16_map
    bl d1_q4_i8_map
    mov x0,#0
    LEAVE
.size d1_backbone_panel_loader,.-d1_backbone_panel_loader

.type d1_join_backbone_panels,%function
d1_join_backbone_panels:
    ADDR x0,d1_async_loader_active
    ldr w0,[x0]
    cbz w0,.Lno_backbone_loader
    ENTER
    ADDR x0,d1_async_join_start
    bl clock_stamp
    ADDR x0,d1_async_loader_thread
    ldr x0,[x0]
    mov x1,#0
    bl pthread_join
    cbnz w0,thread_error
    ADDR x0,d1_async_join_end
    bl clock_stamp
    ADDR x0,d1_async_join_start
    ADDR x1,d1_async_join_end
    bl seconds_between
    ADDR x0,d1_async_join_seconds
    str d0,[x0]
    ADDR x0,d1_async_loader_active
    str wzr,[x0]
    ADDR x0,d1_async_loader_joined
    mov w1,#1
    str w1,[x0]
    LEAVE
.Lno_backbone_loader:
    ret
.size d1_join_backbone_panels,.-d1_join_backbone_panels

.section .rodata
d1_async_loader_env: .asciz "D1_ASYNC_BACKBONE_PANELS"
d1_async_loader_stats: .asciz "d1_async_backbone_panels nice=%d join_s=%.9f\n"
.bss
.balign 8
d1_async_loader_thread: .skip 8
d1_async_join_start: .skip 16
d1_async_join_end: .skip 16
d1_async_join_seconds: .skip 8
d1_async_loader_active: .skip 4
d1_async_loader_joined: .skip 4
d1_async_loader_nice: .skip 4
.text
.endif
