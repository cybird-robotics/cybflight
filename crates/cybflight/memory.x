MEMORY {
    FLASH   : ORIGIN = 0x08000000, LENGTH = 1920K
    RAM     : ORIGIN = 0x24000000, LENGTH = 512K
    /* STM32H7 D3-domain backup SRAM. 4 KiB, byte-addressable, retained
       across soft-reset and (when VBAT is wired) full power loss.
       Used by the post-mortem subsystem; nothing else may place data here. */
    BKPSRAM : ORIGIN = 0x38800000, LENGTH = 4K
}

/* Post-mortem record lives in BKPSRAM. NOLOAD because the contents are
   filled at runtime; the linker only needs to assign the address.
   The fault-handler / pre_init paths reach this static via a raw pointer,
   so we pin a single symbol here rather than letting Rust place it. */
SECTIONS {
    .bkpsram (NOLOAD) : ALIGN(8) {
        *(.bkpsram .bkpsram.*)
    } > BKPSRAM
}
