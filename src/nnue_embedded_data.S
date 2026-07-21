// Embeds the default NNUE network directly into the binary via .incbin, so the
// engine works standalone without the .nnue file sitting next to the executable.
// NNUE_EMBED_PATH is supplied by the Makefile (-DNNUE_EMBED_PATH="...").
// See nnue.cpp's load_embedded_network() for the consumer.
//
// Mach-O (macOS) requires its own segment/section syntax and a leading
// underscore on C symbol names; every other target here (ELF/Linux, PE/COFF
// x86_64-Windows via mingw) uses plain section names and no underscore.
#if defined(__APPLE__)
    .section __TEXT,__const
    .globl _g_embedded_nnue_data
    .globl _g_embedded_nnue_end
_g_embedded_nnue_data:
    .incbin NNUE_EMBED_PATH
_g_embedded_nnue_end:
#else
    .section .rodata
    .globl g_embedded_nnue_data
    .globl g_embedded_nnue_end
g_embedded_nnue_data:
    .incbin NNUE_EMBED_PATH
g_embedded_nnue_end:
#endif

// Mark the object as not requiring an executable stack (GNU/ELF hardening
// convention); hand-written .S files otherwise default to executable-stack-
// required and the linker warns.
#if defined(__ELF__)
.section .note.GNU-stack,"",@progbits
#endif
