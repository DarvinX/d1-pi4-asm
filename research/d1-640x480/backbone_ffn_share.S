/* Share late normalized FFN inputs among protected image rows. */
.arch armv8-a
.include "src/common.inc"
.text
.balign 64
.global d1_share_late_ffn
.global d1_share_late_ffn_weighted
.global d1_share_late_ffn8
.type d1_share_late_ffn,%function
.type d1_share_late_ffn_weighted,%function
.type d1_share_late_ffn8,%function
.ifdef D1_BACKBONE_REDUCED_ROWS
.global d1_share_late_ffn4
.type d1_share_late_ffn4,%function
.endif
/* x0 compact input, x1 sorted active IDs, w2 count (<=512), x3 sorted
 * representative indices in that input, w4 first image ID, x5 full scatter map,
 * s0 minimum cosine similarity.
 * Inactive entries of the scatter map already contain -1. */
d1_share_late_ffn:
    mov w6,#20
    mov w7,#0
    b .Lshare
d1_share_late_ffn8:
    mov w6,#8
    mov w7,#0
    b .Lshare
.ifdef D1_BACKBONE_REDUCED_ROWS
d1_share_late_ffn4:
    mov w6,#4
    mov w7,#0
    b .Lshare
.endif
d1_share_late_ffn_weighted:
    mov w6,#12
    mov w7,#4
.Lshare:
    ENTER
    sub sp,sp,#2144
    fmov s11,w6
    str w7,[sp,#2128]
    mov x19,x0
    mov x20,x1
    mov w21,w2
    mov x22,x3
    mov x23,x5
    mov w24,w4
    add w25,w4,#300
    fmov s10,s0
    mov w26,#0
.Lrep_norms:
    ldr w0,[x22,x26,lsl #2]
    add x0,x19,x0,lsl #13
    mov x1,x0
    mov w2,#2048
    bl lfm_dot_native
    fsqrt s0,s0
    add x8,sp,#2048
    str s0,[x8,x26,lsl #2]
    add w26,w26,#1
    fmov w0,s11
    cmp w26,w0
    b.lo .Lrep_norms

    mov w26,#0
.Lassign:
    str w26,[sp,x26,lsl #2]
    ldr w0,[x20,x26,lsl #2]
    cmp w0,w24
    b.lo .Lassign_next
    cmp w0,w25
    b.hs .Lassign_next
    mov w27,#0
.Lprotected:
    ldr w0,[x22,x27,lsl #2]
    cmp w0,w26
    b.eq .Lassign_next
    add w27,w27,#1
    fmov w0,s11
    cmp w27,w0
    b.lo .Lprotected
    add x0,x19,x26,lsl #13
    mov x1,x0
    mov w2,#2048
    bl lfm_dot_native
    fsqrt s8,s0
    fcmp s8,#0.0
    b.le .Lassign_next
    fmov s9,s10
    mov w27,#0
.Lcompare_rep:
    ldr w28,[x22,x27,lsl #2]
    add x0,x19,x26,lsl #13
    add x1,x19,x28,lsl #13
    mov w2,#2048
    bl lfm_dot_native
    add x8,sp,#2048
    ldr s1,[x8,x27,lsl #2]
    fmul s1,s1,s8
    fcmp s1,#0.0
    b.le .Lcompare_next
    fdiv s0,s0,s1
    fcmp s0,s9
    b.lt .Lcompare_next
    fmov s9,s0
    str w28,[sp,x26,lsl #2]
.Lcompare_next:
    add w27,w27,#1
    fmov w0,s11
    cmp w27,w0
    b.lo .Lcompare_rep
.Lassign_next:
    add w26,w26,#1
    cmp w26,w21
    b.lo .Lassign

    ldr w25,[sp,#2128]
    cbz w25,.Lcompact
    /* Compute centroids only after assignments are final. Each row belongs
       to one group, so a representative may be updated in place. */
    mov w26,#0
.Lcentroid:
    ldr w28,[x22,x26,lsl #2]
    add x8,sp,#2048
    ldr s0,[x8,x26,lsl #2]
    fcvt d0,s0
    ucvtf d9,w25
    fmul d8,d0,d9
    mov w24,w25
    mov w27,#0
.Lcentroid_member:
    ldr w0,[sp,x27,lsl #2]
    cmp w0,w28
    b.ne .Lcentroid_next
    cmp w27,w28
    b.eq .Lcentroid_next
    cmp w24,w25
    b.ne .Lcentroid_add
    add x0,x19,x28,lsl #13
    ucvtf s0,w25
    bl .Lscale_row
.Lcentroid_add:
    add x0,x19,x27,lsl #13
    mov x1,x0
    mov w2,#2048
    bl lfm_dot_native
    fsqrt s0,s0
    fcvt d0,s0
    fadd d8,d8,d0
    add x0,x19,x28,lsl #13
    add x1,x19,x27,lsl #13
    mov w2,#2048
    bl lfm_add
    add w24,w24,#1
.Lcentroid_next:
    add w27,w27,#1
    cmp w27,w21
    b.lo .Lcentroid_member
    cmp w24,w25
    b.eq .Lcentroid_done
    add x0,x19,x28,lsl #13
    mov x1,x0
    mov w2,#2048
    bl lfm_dot_native
    fsqrt s0,s0
    fcvt d0,s0
    ucvtf d1,w24
    fmul d0,d0,d1
    fdiv d0,d8,d0
    fcvt s0,d0
    add x0,x19,x28,lsl #13
    bl .Lscale_row
.Lcentroid_done:
    add w26,w26,#1
    fmov w0,s11
    cmp w26,w0
    b.lo .Lcentroid

.Lcompact:
    /* Resolve every kept row before mapping duplicates. A representative
       can appear later in the original raster than a row it serves. */
    mov w26,#0
    mov w27,#0
.Lcopy_kept:
    ldr w0,[sp,x26,lsl #2]
    cmp w0,w26
    b.ne .Lcopy_next
    cmp w26,w27
    b.eq .Lmap_kept
    add x0,x19,x27,lsl #13
    add x1,x19,x26,lsl #13
    mov x2,#8192
    bl memcpy
.Lmap_kept:
    ldr w0,[x20,x26,lsl #2]
    str w27,[x23,x0,lsl #2]
    add w27,w27,#1
.Lcopy_next:
    add w26,w26,#1
    cmp w26,w21
    b.lo .Lcopy_kept

    mov w26,#0
.Lmap_shared:
    ldr w0,[sp,x26,lsl #2]
    cmp w0,w26
    b.eq .Lmap_next
    ldr w0,[x20,x0,lsl #2]
    ldr w1,[x23,x0,lsl #2]
    ldr w0,[x20,x26,lsl #2]
    str w1,[x23,x0,lsl #2]
.Lmap_next:
    add w26,w26,#1
    cmp w26,w21
    b.lo .Lmap_shared
    mov w0,w27
    add sp,sp,#2144
    LEAVE
.size d1_share_late_ffn,.-d1_share_late_ffn
.size d1_share_late_ffn_weighted,.-d1_share_late_ffn_weighted
.size d1_share_late_ffn8,.-d1_share_late_ffn8
.ifdef D1_BACKBONE_REDUCED_ROWS
.size d1_share_late_ffn4,.-d1_share_late_ffn4
.endif

.Lscale_row:
    dup v0.4s,v0.s[0]
    add x1,x0,#8192
.Lscale_vector:
    ldp q1,q2,[x0]
    fmul v1.4s,v1.4s,v0.4s
    fmul v2.4s,v2.4s,v0.4s
    stp q1,q2,[x0],#32
    cmp x0,x1
    b.ne .Lscale_vector
    ret

.ifdef D1_BACKBONE_TRANSITION_FFN
.global d1_ffn_compact_keep
.type d1_ffn_compact_keep,%function
/* x0 compact 2048-wide inputs, x1 original row-to-representative map,
 * x2 active mask, w3 original rows, w4 compact rows, x5 remap scratch. */
d1_ffn_compact_keep:
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov w22,w3
    mov w23,w4
    mov x24,x5
    mov x0,x24
    mov w1,#-1
    lsl w2,w23,#2
    bl memset
    mov w25,#0
    mov w26,#-1
.Lkeep_mark:
    ldrb w0,[x21,x25]
    cbz w0,.Lkeep_drop
    ldr w0,[x20,x25,lsl #2]
    str wzr,[x24,x0,lsl #2]
    b .Lkeep_next_row
.Lkeep_drop:
    str w26,[x20,x25,lsl #2]
.Lkeep_next_row:
    add w25,w25,#1
    cmp w25,w22
    b.lo .Lkeep_mark

    mov w25,#0
    mov w26,#0
.Lkeep_copy:
    ldr w0,[x24,x25,lsl #2]
    tbnz w0,#31,.Lkeep_next_rep
    str w26,[x24,x25,lsl #2]
    cmp w26,w25
    b.eq .Lkeep_copied
    add x0,x19,x26,lsl #13
    add x1,x19,x25,lsl #13
    mov w2,#8192
    bl memcpy
.Lkeep_copied:
    add w26,w26,#1
.Lkeep_next_rep:
    add w25,w25,#1
    cmp w25,w23
    b.lo .Lkeep_copy

    mov w25,#0
.Lkeep_translate:
    ldr w0,[x20,x25,lsl #2]
    tbnz w0,#31,.Lkeep_translated
    ldr w0,[x24,x0,lsl #2]
    str w0,[x20,x25,lsl #2]
.Lkeep_translated:
    add w25,w25,#1
    cmp w25,w22
    b.lo .Lkeep_translate
    mov w0,w26
    LEAVE
.size d1_ffn_compact_keep,.-d1_ffn_compact_keep
.endif
.section .note.GNU-stack,"",%progbits
