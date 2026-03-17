fn main() {
    // Copy memory.x to OUT_DIR so it overrides embassy-stm32's generated linker script.
    // This shrinks the FLASH region from 2048K to 1920K, reserving sector 7 for params.
    let out = std::env::var("OUT_DIR").unwrap();
    std::fs::copy("memory.x", std::path::Path::new(&out).join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out);

    println!("cargo:rustc-link-arg-bins=--nmagic");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    println!("cargo:rustc-link-arg-bins=-Tdefmt.x");
    println!("cargo:rerun-if-changed=memory.x");
}
