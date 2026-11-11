/* Group neighboring normalized rows without changing the original
 * residuals. Map each original position to its compacted FFN row. */
.macro D1_FFN_COMPACT name,channels,grid_width,grid_height,row_shift,cosine,tile_log=1,centroid=0
.type \name,%function
\name: /* rows, row map, first image row, count -> unique rows */
    ENTER
    sub sp,sp,#(8<<(2*\tile_log))
    mov x19,x0
    mov x20,x1
    mov w21,w2
    mov w22,w3
    .if \channels == 1024 && \grid_width == 40
    .ifdef D1_VISION_FFN_MASKED
    fmov s10,s0
    .else
    ldr s10,.Lreuse_cosine\@
    .endif
    .elseif \channels == 1024 && \grid_width == 20
    .ifdef D1_VISION_LATE_FFN_TUNE
    fmov s10,s0
    .else
    ldr s10,.Lreuse_cosine\@
    .endif
    .elseif \centroid && \channels == 2048
    .ifdef D1_SPARSE_WORK
    ADDR x0,d1_sparse_ffn_cutoff
    ldr s10,[x0]
    .else
    ldr s10,.Lreuse_cosine\@
    .endif
    .else
    ldr s10,.Lreuse_cosine\@
    .endif
    mov w23,#0
.Lreuse_identity\@:
    str w23,[x20,x23,lsl #2]
    add w23,w23,#1
    cmp w23,w22
    b.lo .Lreuse_identity\@
    mov w23,#0
.Lreuse_tile\@:
    mov w0,#(\grid_width>>\tile_log)
    udiv w1,w23,w0
    msub w2,w1,w0,w23
    mov w0,#(\grid_width<<\tile_log)
    madd w24,w1,w0,w21
    add w24,w24,w2,lsl #\tile_log
    mov w25,#0
    mov w26,#0
.Lreuse_row\@:
    lsr w0,w25,#\tile_log
    mov w1,#\grid_width
    madd w27,w0,w1,w24
    and w0,w25,#((1<<\tile_log)-1)
    add w27,w27,w0
    add w0,w21,#(\grid_width*\grid_height)
    cmp w27,w0
    b.hs .Lreuse_next_tile\@
    add x0,x19,x27,lsl #\row_shift
    mov x1,x0
    mov w2,#\channels
    bl lfm_dot_native
    fsqrt s8,s0
    fmov s9,s10
    mov w28,#0
.Lreuse_compare\@:
    cmp w28,w26
    b.hs .Lreuse_keep\@
    ldr s0,[sp,x28,lsl #2]
    fmul s0,s0,s8
    fcmp s0,#0.0
    b.le .Lreuse_next_rep\@
    add x2,sp,#(4<<(2*\tile_log))
    ldr w1,[x2,x28,lsl #2]
    add x0,x19,x27,lsl #\row_shift
    add x1,x19,x1,lsl #\row_shift
    mov w2,#\channels
    bl lfm_dot_native
    ldr s1,[sp,x28,lsl #2]
    fmul s1,s1,s8
    fdiv s0,s0,s1
    fcmp s0,s9
    b.lt .Lreuse_next_rep\@
    fmov s9,s0
    add x0,sp,#(4<<(2*\tile_log))
    ldr w1,[x0,x28,lsl #2]
    str w1,[x20,x27,lsl #2]
.Lreuse_next_rep\@:
    add w28,w28,#1
    b .Lreuse_compare\@
.Lreuse_keep\@:
    ldr w0,[x20,x27,lsl #2]
    cmp w0,w27
    b.ne .Lreuse_next_row\@
    str s8,[sp,x26,lsl #2]
    add x0,sp,#(4<<(2*\tile_log))
    str w27,[x0,x26,lsl #2]
    add w26,w26,#1
.Lreuse_next_row\@:
    add w25,w25,#1
    cmp w25,#(1<<(2*\tile_log))
    b.lo .Lreuse_row\@
.Lreuse_next_tile\@:
    add w23,w23,#1
    cmp w23,#((\grid_width>>\tile_log)*((\grid_height+(1<<\tile_log)-1)>>\tile_log))
    b.lo .Lreuse_tile\@
    .if \centroid
    mov x0,x19
    mov x1,x20
    mov w2,w21
    .if \channels == 2048
    bl d1_restore_ffn_centroids
    .else
    bl d1_restore_vision_ffn_centroids
    .endif
    .endif

    /* First representatives precede each duplicate in raster order.
     * Compact in place, then translate each duplicate's original index. */
    mov w23,#0
    mov w24,#0
.Lreuse_compact\@:
    ldr w0,[x20,x23,lsl #2]
    cmp w0,w23
    b.ne .Lreuse_remap\@
    cmp w24,w23
    b.eq .Lreuse_store\@
    add x0,x19,x24,lsl #\row_shift
    add x1,x19,x23,lsl #\row_shift
    mov x2,#(\channels*4)
    bl memcpy
.Lreuse_store\@:
    mov w0,w24
    add w24,w24,#1
    b .Lreuse_map_done\@
.Lreuse_remap\@:
    ldr w0,[x20,x0,lsl #2]
.Lreuse_map_done\@:
    str w0,[x20,x23,lsl #2]
    add w23,w23,#1
    cmp w23,w22
    b.lo .Lreuse_compact\@
    mov w0,w24
    add sp,sp,#(8<<(2*\tile_log))
    LEAVE
.size \name,.-\name
.balign 4
.Lreuse_cosine\@: .float \cosine
.endm
D1_FFN_COMPACT d1_compact_ffn_rows,2048,20,15,13,0.38
.ifdef D1_VISION_FFN_CENTROID
D1_FFN_COMPACT d1_compact_vision_ffn,1024,40,30,12,0.50,1,1
.else
D1_FFN_COMPACT d1_compact_vision_ffn,1024,40,30,12,0.50
.endif
.ifdef D1_VISION_LATE_FFN
D1_FFN_COMPACT d1_compact_vision_late_ffn,1024,20,15,12,0.80
.endif
.global d1_compact_ffn_centroids
D1_FFN_COMPACT d1_compact_ffn_centroids,2048,20,15,13,0.38,2,1

/* Rows, original representative map and first image row. Each temporary
 * record holds the sum of member norms and the member count. */
.macro D1_RESTORE_FFN_CENTROIDS name,channels,positions,row_shift
.type \name,%function
\name:
    ENTER
    .if \positions == 1200
    sub sp,sp,#2,lsl #12
    .endif
    sub sp,sp,#((8*\positions)%4096)
    mov x19,x0
    mov x20,x1
    mov w21,w2
    add w22,w21,#\positions
    mov w23,w21
    mov x0,sp
    mov w1,#0
    mov w2,#8*\positions
    bl memset
    /* First representatives precede their members. A member's
       original row is intact until its norm and sum are read. */
.Lcentroid_collect\@:
    ldr w24,[x20,x23,lsl #2]
    sub w25,w24,w21
    add x25,sp,x25,lsl #3
    add x26,x19,x23,lsl #\row_shift
    mov x0,x26
    mov x1,x26
    mov w2,#\channels
    bl lfm_dot_native
    fsqrt s0,s0
    ldr s1,[x25]
    fadd s0,s0,s1
    str s0,[x25]
    ldr w0,[x25,#4]
    add w0,w0,#1
    str w0,[x25,#4]
    cmp w24,w23
    b.eq .Lcentroid_next\@
    add x0,x19,x24,lsl #\row_shift
    mov x1,x26
    mov w2,#\channels
    bl lfm_add
.Lcentroid_next\@:
    add w23,w23,#1
    cmp w23,w22
    b.lo .Lcentroid_collect\@
    mov w23,w21
    mov x25,sp
.Lcentroid_normalize\@:
    ldr w24,[x25,#4]
    cmp w24,#2
    b.lo .Lcentroid_skip\@
    add x26,x19,x23,lsl #\row_shift
    mov x0,x26
    mov x1,x26
    mov w2,#\channels
    bl lfm_dot_native
    fcmp s0,#0.0
    b.le .Lcentroid_skip\@
    fsqrt s0,s0
    ucvtf s1,w24
    fmul s0,s0,s1
    ldr s1,[x25]
    fdiv s0,s1,s0
    mov w0,#\channels/4
.Lcentroid_scale\@:
    ldr q1,[x26]
    fmul v1.4s,v1.4s,v0.s[0]
    str q1,[x26],#16
    subs w0,w0,#1
    b.ne .Lcentroid_scale\@
.Lcentroid_skip\@:
    add x25,x25,#8
    add w23,w23,#1
    cmp w23,w22
    b.lo .Lcentroid_normalize\@
    add sp,sp,#((8*\positions)%4096)
    .if \positions == 1200
    add sp,sp,#2,lsl #12
    .endif
    LEAVE
.size \name,.-\name
.endm

D1_RESTORE_FFN_CENTROIDS d1_restore_ffn_centroids,2048,300,13
.ifdef D1_VISION_FFN_CENTROID
D1_RESTORE_FFN_CENTROIDS d1_restore_vision_ffn_centroids,1024,1200,12
.endif
