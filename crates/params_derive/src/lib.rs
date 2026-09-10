//! `#[derive(Params)]` — generates `cybflight_core::param_registry::ParamGroup`.
//!
//! Field attributes (`#[param(...)]`):
//! - *(none)* — scalar param named after the field. Types: `f32`, `u8`,
//!   `u16`, `bool`.
//! - `key = "name"` — rename a scalar param.
//! - `keys = "a,b,c"` — required for `[f32; N]` / `[[f32; C]; R]` fields;
//!   comma-separated full names, one per element (row-major for 2D).
//! - `nested` — field type is itself a `ParamGroup`; its params are
//!   spliced in under their own names.
//! - `nested_array, prefix = "m"` — `[T; N]` where `T: ParamGroup`;
//!   element `i`'s params are exposed as `m{i}_{name}`.
//! - `as_u16` — expose a `usize` field as a u16 param (saturating).
//! - `enum_u8` — field type implements `ParamEnum`; exposed as u8.
//! - `skip` — field is not a parameter.
//! - `reboot` — marks the param (or, on `nested`/`nested_array` fields,
//!   every delegated param) as taking effect only after a reboot.
//! - `unit = "..."`, `min = ...`, `max = ...` — optional metadata surfaced
//!   via `ParamGroup::param_meta` (shell range validation, generated docs).
//!   Array fields share one entry; the field's first doc-comment line is
//!   captured as the description.
//!
//! Index order is declaration order, depth-first. `COUNT` is the total
//! number of scalar params.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input, spanned::Spanned};

enum FieldKind {
    Skip,
    Scalar(ScalarTy, String),
    /// (keys, rows, cols) — 1D arrays use rows == 1, cols == N,
    /// 2D `[[f32; C]; R]` uses (R, C). Keys are row-major.
    ArrayF32(Vec<String>, usize, usize),
    Nested(Type),
    NestedArray(Type, usize, String),
}

/// Optional per-field metadata from `#[param(unit/min/max)]` + doc comments.
#[derive(Default)]
struct MetaSpec {
    unit: Option<String>,
    min: Option<f32>,
    max: Option<f32>,
    doc: Option<String>,
    reboot: bool,
}

impl MetaSpec {
    fn to_tokens(&self) -> TokenStream2 {
        let unit = self.unit.clone().unwrap_or_default();
        let doc = self.doc.clone().unwrap_or_default();
        let min = match self.min {
            Some(v) => quote!(#v),
            None => quote!(::core::f32::NEG_INFINITY),
        };
        let max = match self.max {
            Some(v) => quote!(#v),
            None => quote!(::core::f32::INFINITY),
        };
        let reboot = self.reboot;
        quote!(::cybflight_core::param_registry::ParamMeta {
            unit: #unit,
            min: #min,
            max: #max,
            doc: #doc,
            reboot: #reboot,
        })
    }
}

/// Parse a (possibly negated) numeric literal expression to f32.
fn parse_f32_expr(expr: &syn::Expr) -> Option<f32> {
    match expr {
        syn::Expr::Lit(l) => match &l.lit {
            syn::Lit::Float(f) => f.base10_parse().ok(),
            syn::Lit::Int(i) => i.base10_parse::<f32>().ok(),
            _ => None,
        },
        syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => {
            parse_f32_expr(&u.expr).map(|v| -v)
        }
        _ => None,
    }
}

/// First non-empty line of the field's `///` doc comment.
fn doc_line(field: &syn::Field) -> Option<String> {
    for attr in &field.attrs {
        if attr.path().is_ident("doc")
            && let syn::Meta::NameValue(nv) = &attr.meta
            && let syn::Expr::Lit(l) = &nv.value
            && let syn::Lit::Str(ls) = &l.lit
        {
            let line = ls.value().trim().to_string();
            if !line.is_empty() {
                return Some(line);
            }
        }
    }
    None
}

enum ScalarTy {
    F32,
    U8,
    U16,
    U32,
    Bool,
    UsizeAsU16,
    EnumU8,
}

fn array_len(len: &syn::Expr) -> Option<usize> {
    if let syn::Expr::Lit(l) = len
        && let syn::Lit::Int(i) = &l.lit
    {
        return i.base10_parse().ok();
    }
    None
}

fn classify(field: &syn::Field) -> Result<(FieldKind, MetaSpec), syn::Error> {
    let mut key: Option<String> = None;
    let mut keys: Option<Vec<String>> = None;
    let mut nested = false;
    let mut nested_array = false;
    let mut prefix: Option<String> = None;
    let mut as_u16 = false;
    let mut enum_u8 = false;
    let mut skip = false;
    let mut meta = MetaSpec {
        doc: doc_line(field),
        ..MetaSpec::default()
    };
    let mut unit_attr: Option<String> = None;
    let mut reboot_attr = false;
    let mut min_attr: Option<f32> = None;
    let mut max_attr: Option<f32> = None;

    for attr in &field.attrs {
        if !attr.path().is_ident("param") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("key") {
                key = Some(meta.value()?.parse::<syn::LitStr>()?.value());
            } else if meta.path.is_ident("keys") {
                let s = meta.value()?.parse::<syn::LitStr>()?.value();
                keys = Some(s.split(',').map(|k| k.trim().to_string()).collect());
            } else if meta.path.is_ident("nested") {
                nested = true;
            } else if meta.path.is_ident("nested_array") {
                nested_array = true;
            } else if meta.path.is_ident("prefix") {
                prefix = Some(meta.value()?.parse::<syn::LitStr>()?.value());
            } else if meta.path.is_ident("as_u16") {
                as_u16 = true;
            } else if meta.path.is_ident("enum_u8") {
                enum_u8 = true;
            } else if meta.path.is_ident("skip") {
                skip = true;
            } else if meta.path.is_ident("reboot") {
                reboot_attr = true;
            } else if meta.path.is_ident("unit") {
                unit_attr = Some(meta.value()?.parse::<syn::LitStr>()?.value());
            } else if meta.path.is_ident("min") {
                let expr = meta.value()?.parse::<syn::Expr>()?;
                min_attr = Some(
                    parse_f32_expr(&expr)
                        .ok_or_else(|| meta.error("min must be a numeric literal"))?,
                );
            } else if meta.path.is_ident("max") {
                let expr = meta.value()?.parse::<syn::Expr>()?;
                max_attr = Some(
                    parse_f32_expr(&expr)
                        .ok_or_else(|| meta.error("max must be a numeric literal"))?,
                );
            } else {
                return Err(meta.error("unknown #[param] attribute"));
            }
            Ok(())
        })?;
    }

    meta.unit = unit_attr;
    meta.min = min_attr;
    meta.max = max_attr;
    meta.reboot = reboot_attr;

    if skip {
        return Ok((FieldKind::Skip, meta));
    }
    let err = |msg: &str| Err(syn::Error::new(field.span(), msg));

    if nested {
        return Ok((FieldKind::Nested(field.ty.clone()), meta));
    }
    if nested_array {
        let Type::Array(arr) = &field.ty else {
            return err("#[param(nested_array)] requires a [T; N] field");
        };
        let Some(n) = array_len(&arr.len) else {
            return err("nested_array length must be an integer literal");
        };
        let Some(prefix) = prefix else {
            return err("#[param(nested_array)] requires prefix = \"...\"");
        };
        return Ok((FieldKind::NestedArray((*arr.elem).clone(), n, prefix), meta));
    }

    match &field.ty {
        Type::Array(arr) => {
            let Some(keys) = keys else {
                return err("array field needs #[param(keys = \"...\")] (or nested_array/skip)");
            };
            let Some(n) = array_len(&arr.len) else {
                return err("array length must be an integer literal");
            };
            let (rows, cols) = match &*arr.elem {
                Type::Array(inner) => {
                    let Some(c) = array_len(&inner.len) else {
                        return err("inner array length must be an integer literal");
                    };
                    (n, c)
                }
                _ => (1, n),
            };
            if keys.len() != rows * cols {
                return err(&format!(
                    "keys list has {} entries but the array has {} elements",
                    keys.len(),
                    rows * cols
                ));
            }
            Ok((FieldKind::ArrayF32(keys, rows, cols), meta))
        }
        ty => {
            let name = key.unwrap_or_else(|| field.ident.as_ref().unwrap().to_string());
            let scalar = if enum_u8 {
                ScalarTy::EnumU8
            } else if as_u16 {
                ScalarTy::UsizeAsU16
            } else {
                let Type::Path(p) = ty else {
                    return err("unsupported param field type");
                };
                match p.path.segments.last().map(|s| s.ident.to_string()).as_deref() {
                    Some("f32") => ScalarTy::F32,
                    Some("u8") => ScalarTy::U8,
                    Some("u16") => ScalarTy::U16,
                    Some("u32") => ScalarTy::U32,
                    Some("bool") => ScalarTy::Bool,
                    Some("usize") => {
                        return err("usize fields need #[param(as_u16)] (or skip)");
                    }
                    _ => {
                        return err(
                            "unsupported scalar type; use nested/enum_u8/skip for non-primitives",
                        );
                    }
                }
            };
            Ok((FieldKind::Scalar(scalar, name), meta))
        }
    }
}

#[proc_macro_derive(Params, attributes(param))]
pub fn derive_params(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let Data::Struct(data) = &input.data else {
        return syn::Error::new(input.span(), "#[derive(Params)] supports structs only")
            .to_compile_error()
            .into();
    };
    let Fields::Named(fields) = &data.fields else {
        return syn::Error::new(input.span(), "#[derive(Params)] requires named fields")
            .to_compile_error()
            .into();
    };

    let mut count_terms: Vec<TokenStream2> = Vec::new();
    let mut get_arms: Vec<TokenStream2> = Vec::new();
    let mut set_arms: Vec<TokenStream2> = Vec::new();
    let mut find_arms: Vec<TokenStream2> = Vec::new();
    let mut name_arms: Vec<TokenStream2> = Vec::new();
    let mut path_arms: Vec<TokenStream2> = Vec::new();

    let mut meta_arms: Vec<TokenStream2> = Vec::new();
    for field in &fields.named {
        let (kind, mspec) = match classify(field) {
            Ok(k) => k,
            Err(e) => return e.to_compile_error().into(),
        };
        let meta_tokens = mspec.to_tokens();
        let ident = field.ident.as_ref().unwrap();
        // Field name as a literal — the group-path segment contributed by a
        // nested field (see `ParamGroup::param_write_path`).
        let ident_str = ident.to_string();
        let reg = quote!(::cybflight_core::param_registry);

        match kind {
            FieldKind::Skip => {
                let _ = meta_tokens;
            }
            FieldKind::Scalar(ty, key) => {
                count_terms.push(quote!(1));
                let (get_val, set_stmt) = match ty {
                    ScalarTy::F32 => (
                        quote!(#reg::ParamValue::F32(self.#ident)),
                        quote!(self.#ident = v;),
                    ),
                    ScalarTy::U8 => (
                        quote!(#reg::ParamValue::U8(self.#ident)),
                        quote!(self.#ident = #reg::f32_to_u8(v);),
                    ),
                    ScalarTy::U16 => (
                        quote!(#reg::ParamValue::U16(self.#ident)),
                        quote!(self.#ident = #reg::f32_to_u16(v);),
                    ),
                    ScalarTy::U32 => (
                        quote!(#reg::ParamValue::U32(self.#ident)),
                        quote!(self.#ident = #reg::f32_to_u32(v);),
                    ),
                    ScalarTy::Bool => (
                        quote!(#reg::ParamValue::Bool(self.#ident)),
                        quote!(self.#ident = #reg::f32_to_bool(v);),
                    ),
                    ScalarTy::UsizeAsU16 => (
                        quote!(#reg::ParamValue::U16(#reg::f32_to_u16(self.#ident as f32))),
                        quote!(self.#ident = #reg::f32_to_u16(v) as usize;),
                    ),
                    ScalarTy::EnumU8 => (
                        quote!(#reg::ParamValue::U8(#reg::ParamEnum::to_u8(self.#ident))),
                        quote!(self.#ident = #reg::ParamEnum::from_u8(#reg::f32_to_u8(v));),
                    ),
                };
                get_arms.push(quote! {
                    if i == 0 { return Some(#get_val); }
                    i -= 1;
                });
                set_arms.push(quote! {
                    if i == 0 { #set_stmt return true; }
                    i -= 1;
                });
                find_arms.push(quote! {
                    if name == #key { return Some(base); }
                    base += 1;
                });
                name_arms.push(quote! {
                    if i == 0 { return w.write_str(#key); }
                    i -= 1;
                });
                path_arms.push(quote! {
                    if i == 0 { return Ok(()); }
                    i -= 1;
                });
                meta_arms.push(quote! {
                    if i == 0 { return #meta_tokens; }
                    i -= 1;
                });
            }
            FieldKind::ArrayF32(keys, rows, cols) => {
                let n = rows * cols;
                count_terms.push(quote!(#n));
                let elem = if rows == 1 {
                    quote!(self.#ident[i])
                } else {
                    quote!(self.#ident[i / #cols][i % #cols])
                };
                get_arms.push(quote! {
                    if i < #n { return Some(#reg::ParamValue::F32(#elem)); }
                    i -= #n;
                });
                let elem_mut = if rows == 1 {
                    quote!(self.#ident[i] = v;)
                } else {
                    quote!(self.#ident[i / #cols][i % #cols] = v;)
                };
                set_arms.push(quote! {
                    if i < #n { #elem_mut return true; }
                    i -= #n;
                });
                let key_strs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
                let idxs: Vec<usize> = (0..n).collect();
                find_arms.push(quote! {
                    match name {
                        #(#key_strs => return Some(base + #idxs),)*
                        _ => {}
                    }
                    base += #n;
                });
                name_arms.push(quote! {
                    if i < #n {
                        const KEYS: [&str; #n] = [#(#key_strs),*];
                        return w.write_str(KEYS[i]);
                    }
                    i -= #n;
                });
                path_arms.push(quote! {
                    if i < #n { return Ok(()); }
                    i -= #n;
                });
                meta_arms.push(quote! {
                    if i < #n { return #meta_tokens; }
                    i -= #n;
                });
            }
            FieldKind::Nested(ty) => {
                count_terms.push(quote!(<#ty as #reg::ParamGroup>::COUNT));
                get_arms.push(quote! {
                    if i < <#ty as #reg::ParamGroup>::COUNT {
                        return #reg::ParamGroup::param_get(&self.#ident, i);
                    }
                    i -= <#ty as #reg::ParamGroup>::COUNT;
                });
                set_arms.push(quote! {
                    if i < <#ty as #reg::ParamGroup>::COUNT {
                        return #reg::ParamGroup::param_set_f32(&mut self.#ident, i, v);
                    }
                    i -= <#ty as #reg::ParamGroup>::COUNT;
                });
                find_arms.push(quote! {
                    if let Some(j) = <#ty as #reg::ParamGroup>::param_find(name) {
                        return Some(base + j);
                    }
                    base += <#ty as #reg::ParamGroup>::COUNT;
                });
                name_arms.push(quote! {
                    if i < <#ty as #reg::ParamGroup>::COUNT {
                        return <#ty as #reg::ParamGroup>::param_write_name(i, w);
                    }
                    i -= <#ty as #reg::ParamGroup>::COUNT;
                });
                path_arms.push(quote! {
                    if i < <#ty as #reg::ParamGroup>::COUNT {
                        w.write_str(#ident_str)?;
                        // Only emit the separator when the child actually
                        // contributes a segment, so a leaf-bearing subgroup
                        // reads `eskf.filter`, not `eskf.filter.`.
                        let mut probe = #reg::PathProbe::default();
                        <#ty as #reg::ParamGroup>::param_write_path(i, &mut probe)?;
                        if probe.wrote {
                            w.write_str(".")?;
                            return <#ty as #reg::ParamGroup>::param_write_path(i, w);
                        }
                        return Ok(());
                    }
                    i -= <#ty as #reg::ParamGroup>::COUNT;
                });
                let reboot_override = mspec.reboot;
                meta_arms.push(quote! {
                    if i < <#ty as #reg::ParamGroup>::COUNT {
                        let mut m = <#ty as #reg::ParamGroup>::param_meta(i);
                        m.reboot |= #reboot_override;
                        return m;
                    }
                    i -= <#ty as #reg::ParamGroup>::COUNT;
                });
            }
            FieldKind::NestedArray(ty, n, prefix) => {
                count_terms.push(quote!(#n * <#ty as #reg::ParamGroup>::COUNT));
                get_arms.push(quote! {
                    if i < #n * <#ty as #reg::ParamGroup>::COUNT {
                        let sub = <#ty as #reg::ParamGroup>::COUNT;
                        return #reg::ParamGroup::param_get(&self.#ident[i / sub], i % sub);
                    }
                    i -= #n * <#ty as #reg::ParamGroup>::COUNT;
                });
                set_arms.push(quote! {
                    if i < #n * <#ty as #reg::ParamGroup>::COUNT {
                        let sub = <#ty as #reg::ParamGroup>::COUNT;
                        return #reg::ParamGroup::param_set_f32(&mut self.#ident[i / sub], i % sub, v);
                    }
                    i -= #n * <#ty as #reg::ParamGroup>::COUNT;
                });
                find_arms.push(quote! {
                    if let Some(rest) = name.strip_prefix(#prefix) {
                        let digits_end = rest
                            .find(|c: char| !c.is_ascii_digit())
                            .unwrap_or(rest.len());
                        if digits_end > 0
                            && let Some(sub_name) = rest[digits_end..].strip_prefix('_')
                            && let Ok(k) = rest[..digits_end].parse::<usize>()
                            && k < #n
                            && let Some(j) = <#ty as #reg::ParamGroup>::param_find(sub_name)
                        {
                            return Some(base + k * <#ty as #reg::ParamGroup>::COUNT + j);
                        }
                    }
                    base += #n * <#ty as #reg::ParamGroup>::COUNT;
                });
                name_arms.push(quote! {
                    if i < #n * <#ty as #reg::ParamGroup>::COUNT {
                        let sub = <#ty as #reg::ParamGroup>::COUNT;
                        ::core::write!(w, "{}{}_", #prefix, i / sub)?;
                        return <#ty as #reg::ParamGroup>::param_write_name(i % sub, w);
                    }
                    i -= #n * <#ty as #reg::ParamGroup>::COUNT;
                });
                path_arms.push(quote! {
                    if i < #n * <#ty as #reg::ParamGroup>::COUNT {
                        // All array elements share one group: the element
                        // index is already in the name (`m0_`, `m1_`), and
                        // splitting them into per-element groups would make
                        // the listing longer without telling anyone more.
                        return w.write_str(#ident_str);
                    }
                    i -= #n * <#ty as #reg::ParamGroup>::COUNT;
                });
                let reboot_override = mspec.reboot;
                meta_arms.push(quote! {
                    if i < #n * <#ty as #reg::ParamGroup>::COUNT {
                        let mut m = <#ty as #reg::ParamGroup>::param_meta(i % <#ty as #reg::ParamGroup>::COUNT);
                        m.reboot |= #reboot_override;
                        return m;
                    }
                    i -= #n * <#ty as #reg::ParamGroup>::COUNT;
                });
            }
        }
    }

    if count_terms.is_empty() {
        count_terms.push(quote!(0));
    }

    let reg = quote!(::cybflight_core::param_registry);
    let expanded = quote! {
        impl #reg::ParamGroup for #name {
            const COUNT: usize = #(#count_terms)+*;

            fn param_get(&self, idx: usize) -> Option<#reg::ParamValue> {
                let mut i = idx;
                #(#get_arms)*
                let _ = i;
                None
            }

            fn param_set_f32(&mut self, idx: usize, v: f32) -> bool {
                let mut i = idx;
                let _ = v;
                #(#set_arms)*
                let _ = i;
                false
            }

            fn param_meta(idx: usize) -> #reg::ParamMeta {
                let mut i = idx;
                #(#meta_arms)*
                let _ = i;
                #reg::ParamMeta::UNSPEC
            }

            fn param_find(name: &str) -> Option<usize> {
                let mut base = 0usize;
                #(#find_arms)*
                let _ = base;
                None
            }

            fn param_write_name(idx: usize, w: &mut dyn ::core::fmt::Write) -> ::core::fmt::Result {
                let mut i = idx;
                #(#name_arms)*
                let _ = (i, w);
                Err(::core::fmt::Error)
            }

            fn param_write_path(idx: usize, w: &mut dyn ::core::fmt::Write) -> ::core::fmt::Result {
                let mut i = idx;
                #(#path_arms)*
                let _ = (i, w);
                Err(::core::fmt::Error)
            }
        }
    };
    expanded.into()
}
