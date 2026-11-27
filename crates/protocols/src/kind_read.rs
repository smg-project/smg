//! The read of a list whose items are objects of several kinds, the kind named
//! by one key: today the `messages` of a chat request, by `role`. Inside
//! [`direct`] an item is read straight into the variant its kind names; a value
//! that stands before that key is kept as it was parsed until the kind is
//! known. An item that is no object and every read outside [`direct`] take the
//! derived read of the type, which first copies the whole item into a generic
//! tree. Both read a body into the same value and refuse the same bodies: a
//! value the variant ignores is parsed before it is dropped, as the derived
//! read parses it. Every refusal is worded by the derived read.

use std::{borrow::Cow, cell::Cell, fmt, marker::PhantomData};

use serde::{
    de::{
        self,
        value::{
            BorrowedStrDeserializer, MapAccessDeserializer, MapDeserializer, SeqAccessDeserializer,
            SeqDeserializer, StrDeserializer,
        },
        DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess, VariantAccess,
        Visitor,
    },
    Deserialize, Deserializer,
};

thread_local! {
    static DIRECT: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f`, the parse of a body of type `T`, with the direct read of a
/// message on when `T` is the chat request itself. The caller parses again
/// outside when `f` refuses a body, so that the refusal names the path and the
/// place of the error as the derived read does. A chat request nested in
/// another body type may be parsed where a refusal is not retried, so it keeps
/// the derived read: every other type turns the switch off.
#[cfg(feature = "axum")]
pub(crate) fn direct<T: 'static, R>(f: impl FnOnce() -> R) -> R {
    /// Puts the switch back when the direct read ends, also by a panic.
    struct Restore(bool);

    impl Drop for Restore {
        fn drop(&mut self) {
            DIRECT.set(self.0);
        }
    }

    let on =
        std::any::TypeId::of::<T>() == std::any::TypeId::of::<crate::chat::ChatCompletionRequest>();
    let _restore = Restore(DIRECT.replace(on));
    f()
}

pub(crate) fn list<'de, T: ByKind + Deserialize<'de>, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<T>, D::Error> {
    if DIRECT.get() {
        deserializer.deserialize_seq(List(PhantomData))
    } else {
        Vec::deserialize(deserializer)
    }
}

/// A type whose kind is named by one key of its object: the key, the kind as
/// the derived read words it, and the read of one kind from the other keys.
/// `by_kind!` writes it from the type's own definition.
pub(crate) trait ByKind: Sized {
    /// The key that names the kind.
    const KEY: &'static str;
    /// What the derived read says it expects.
    const EXPECTING: &'static str;
    /// The kind, read from the value of the key.
    type Kind: for<'de> Deserialize<'de>;
    /// The value of that kind from the other keys of the object.
    fn of_kind<'de, D: Deserializer<'de>>(kind: Self::Kind, rest: D) -> Result<Self, D::Error>;
}

/// What stands before the kind key in an item: each key and its value.
type Before<'de> = Vec<(Cow<'de, str>, Held<'de>)>;

struct List<T>(PhantomData<T>);

impl<'de, T: ByKind + Deserialize<'de>> Visitor<'de> for List<T> {
    type Value = Vec<T>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a sequence")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut items = Vec::new();
        let mut before = Vec::new();
        while let Some(item) = seq.next_element_seed(Item(&mut before, PhantomData))? {
            items.push(item);
        }
        Ok(items)
    }
}

struct Item<'a, 'de, T>(&'a mut Before<'de>, PhantomData<T>);

impl<'de, T: ByKind + Deserialize<'de>> DeserializeSeed<'de> for Item<'_, 'de, T> {
    type Value = T;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<T, D::Error> {
        // What the item before this one kept is dropped here.
        self.0.clear();
        deserializer.deserialize_any(self)
    }
}

impl<'de, T: ByKind + Deserialize<'de>> Visitor<'de> for Item<'_, 'de, T> {
    type Value = T;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(T::EXPECTING)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<T, A::Error> {
        T::deserialize(SeqAccessDeserializer::new(seq))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<T, A::Error> {
        let mut key = map.next_key_seed(Key)?;
        let before: &mut [(Cow<'de, str>, Held<'de>)] = if key.as_deref() == Some(T::KEY) {
            &mut []
        } else {
            let before = self.0;
            loop {
                match key {
                    None => return Err(de::Error::missing_field(T::KEY)),
                    Some(key) if key == T::KEY => break,
                    Some(key) => before.push((key, map.next_value_seed(Hold)?)),
                }
                key = map.next_key_seed(Key)?;
            }
            before.as_mut_slice()
        };
        let kind = map.next_value_seed(PhantomData::<T::Kind>)?;
        T::of_kind(
            kind,
            MapAccessDeserializer::new(Rest {
                before,
                at: 0,
                held: false,
                of: PhantomData::<T>,
                map,
            }),
        )
    }
}

/// A key of an item, borrowed from the body where it holds no escape.
struct Key;

impl<'de> DeserializeSeed<'de> for Key {
    type Value = Cow<'de, str>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for Key {
    type Value = Cow<'de, str>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a string key")
    }

    fn visit_borrowed_str<E: de::Error>(self, v: &'de str) -> Result<Self::Value, E> {
        Ok(Cow::Borrowed(v))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(Cow::Owned(v.to_owned()))
    }
}

fn hand_key<'de, K: DeserializeSeed<'de>, E: de::Error>(
    seed: K,
    key: Cow<'de, str>,
) -> Result<K::Value, E> {
    match key {
        Cow::Borrowed(key) => seed.deserialize(BorrowedStrDeserializer::new(key)),
        Cow::Owned(key) => seed.deserialize(key.into_deserializer()),
    }
}

/// The keys of an item but the one that names its kind: what stood before it,
/// then what follows. A second such key is refused as the derived read refuses it.
struct Rest<'a, 'de, A, T> {
    before: &'a mut [(Cow<'de, str>, Held<'de>)],
    at: usize,
    held: bool,
    of: PhantomData<T>,
    map: A,
}

impl<'de, A: MapAccess<'de>, T: ByKind> MapAccess<'de> for Rest<'_, 'de, A, T> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
        if let Some((key, _)) = self.before.get(self.at) {
            self.held = true;
            return match *key {
                Cow::Borrowed(key) => seed.deserialize(BorrowedStrDeserializer::new(key)),
                Cow::Owned(ref key) => seed.deserialize(StrDeserializer::new(key)),
            }
            .map(Some);
        }
        match self.map.next_key_seed(Key)? {
            None => Ok(None),
            Some(key) if key == T::KEY => Err(de::Error::duplicate_field(T::KEY)),
            Some(key) => hand_key(seed, key).map(Some),
        }
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, A::Error> {
        if !self.held {
            return self.map.next_value_seed(Strict(seed));
        }
        self.held = false;
        let value = self
            .before
            .get_mut(self.at)
            .map(|(_, value)| std::mem::take(value));
        self.at += 1;
        seed.deserialize(Replayed(value.unwrap_or_default(), PhantomData))
    }
}

/// A value that stood before the kind key, kept as it was parsed until the kind
/// is known.
#[derive(Default)]
enum Held<'de> {
    #[default]
    Null,
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    Str(Cow<'de, str>),
    Seq(Vec<Held<'de>>),
    Map(Vec<(Cow<'de, str>, Held<'de>)>),
}

struct Hold;

impl<'de> DeserializeSeed<'de> for Hold {
    type Value = Held<'de>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Held<'de>, D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Hold {
    type Value = Held<'de>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Held<'de>, E> {
        Ok(Held::Null)
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Held<'de>, E> {
        Ok(Held::Bool(v))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Held<'de>, E> {
        Ok(Held::U64(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Held<'de>, E> {
        Ok(Held::I64(v))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Held<'de>, E> {
        Ok(Held::F64(v))
    }

    fn visit_borrowed_str<E: de::Error>(self, v: &'de str) -> Result<Held<'de>, E> {
        Ok(Held::Str(Cow::Borrowed(v)))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Held<'de>, E> {
        Ok(Held::Str(Cow::Owned(v.to_owned())))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Held<'de>, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(Hold)? {
            items.push(item);
        }
        Ok(Held::Seq(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Held<'de>, A::Error> {
        let mut entries = Vec::new();
        while let Some(key) = map.next_key_seed(Key)? {
            entries.push((key, map.next_value_seed(Hold)?));
        }
        Ok(Held::Map(entries))
    }
}

/// A kept value as a source for the hints the fields of an item ask for: each
/// takes the kinds the JSON parser takes for it. Every other hint and every
/// other kind is refused, and a refused body is read again by the derived read.
struct Replayed<'de, E>(Held<'de>, PhantomData<E>);

impl<'de, E: de::Error> IntoDeserializer<'de, E> for Held<'de> {
    type Deserializer = Replayed<'de, E>;

    fn into_deserializer(self) -> Replayed<'de, E> {
        Replayed(self, PhantomData)
    }
}

fn other_kind<E: de::Error>() -> E {
    E::custom("a value of another kind")
}

macro_rules! kinds {
    ($($method:ident: $($kind:ident)|+;)*) => {$(
        fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, E> {
            match self.0 {
                $(Held::$kind { .. })|+ => self.deserialize_any(visitor),
                _ => Err(other_kind()),
            }
        }
    )*};
}

macro_rules! refused {
    ($($method:ident)*) => {$(
        fn $method<V: Visitor<'de>>(self, _: V) -> Result<V::Value, E> {
            Err(other_kind())
        }
    )*};
}

impl<'de, E: de::Error> Deserializer<'de> for Replayed<'de, E> {
    type Error = E;

    kinds! {
        deserialize_bool: Bool;
        deserialize_i8: U64 | I64 | F64;
        deserialize_i16: U64 | I64 | F64;
        deserialize_i32: U64 | I64 | F64;
        deserialize_i64: U64 | I64 | F64;
        deserialize_u8: U64 | I64 | F64;
        deserialize_u16: U64 | I64 | F64;
        deserialize_u32: U64 | I64 | F64;
        deserialize_u64: U64 | I64 | F64;
        deserialize_f32: U64 | I64 | F64;
        deserialize_f64: U64 | I64 | F64;
        deserialize_str: Str;
        deserialize_string: Str;
        deserialize_identifier: Str;
        deserialize_seq: Seq;
        deserialize_map: Map;
    }

    // No field of an item asks for these: they are refused, not modelled.
    refused! {
        deserialize_i128 deserialize_u128 deserialize_char deserialize_bytes
        deserialize_byte_buf deserialize_unit
    }

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, E> {
        match self.0 {
            Held::Null => visitor.visit_unit(),
            Held::Bool(v) => visitor.visit_bool(v),
            Held::U64(v) => visitor.visit_u64(v),
            Held::I64(v) => visitor.visit_i64(v),
            Held::F64(v) => visitor.visit_f64(v),
            Held::Str(Cow::Borrowed(v)) => visitor.visit_borrowed_str(v),
            Held::Str(Cow::Owned(v)) => visitor.visit_string(v),
            Held::Seq(items) => {
                let mut items = SeqDeserializer::<_, E>::new(items.into_iter());
                let value = visitor.visit_seq(&mut items)?;
                items.end()?;
                Ok(value)
            }
            Held::Map(entries) => {
                let mut entries = MapDeserializer::<_, E>::new(entries.into_iter());
                let value = visitor.visit_map(&mut entries)?;
                entries.end()?;
                Ok(value)
            }
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, E> {
        match self.0 {
            Held::Null => visitor.visit_none(),
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, E> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, E> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_tuple<V: Visitor<'de>>(self, _: usize, _: V) -> Result<V::Value, E> {
        Err(other_kind())
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: usize,
        _: V,
    ) -> Result<V::Value, E> {
        Err(other_kind())
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, E> {
        match self.0 {
            Held::Seq(_) | Held::Map(_) => self.deserialize_any(visitor),
            _ => Err(other_kind()),
        }
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        _: V,
    ) -> Result<V::Value, E> {
        Err(other_kind())
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, E> {
        visitor.visit_unit()
    }
}

/// Hands a value on as it is. A value the reader ignores is parsed before it
/// is dropped, as the derived read parses it, where the JSON parser skips it.
struct Strict<T>(T);

impl<'de, T: DeserializeSeed<'de>> DeserializeSeed<'de> for Strict<T> {
    type Value = T::Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<T::Value, D::Error> {
        self.0.deserialize(Strict(deserializer))
    }
}

macro_rules! hints {
    ($($method:ident)*) => {$(
        fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, D::Error> {
            self.0.$method(Strict(visitor))
        }
    )*};
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for Strict<D> {
    type Error = D::Error;

    hints! {
        deserialize_any deserialize_bool deserialize_i8 deserialize_i16 deserialize_i32
        deserialize_i64 deserialize_i128 deserialize_u8 deserialize_u16 deserialize_u32
        deserialize_u64 deserialize_u128 deserialize_f32 deserialize_f64 deserialize_char
        deserialize_str deserialize_string deserialize_bytes deserialize_byte_buf
        deserialize_option deserialize_unit deserialize_seq deserialize_map
        deserialize_identifier
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.0.deserialize_unit_struct(name, Strict(visitor))
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.0.deserialize_newtype_struct(name, Strict(visitor))
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.0.deserialize_tuple(len, Strict(visitor))
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.0.deserialize_tuple_struct(name, len, Strict(visitor))
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.0.deserialize_struct(name, fields, Strict(visitor))
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, D::Error> {
        self.0.deserialize_enum(name, variants, Strict(visitor))
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, D::Error> {
        self.0.deserialize_any(Checked)?;
        visitor.visit_unit()
    }

    fn is_human_readable(&self) -> bool {
        self.0.is_human_readable()
    }
}

macro_rules! visits {
    ($($method:ident: $kind:ty;)*) => {$(
        fn $method<E: de::Error>(self, v: $kind) -> Result<V::Value, E> {
            self.0.$method(v)
        }
    )*};
}

impl<'de, V: Visitor<'de>> Visitor<'de> for Strict<V> {
    type Value = V::Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.expecting(f)
    }

    visits! {
        visit_bool: bool;
        visit_i8: i8;
        visit_i16: i16;
        visit_i32: i32;
        visit_i64: i64;
        visit_i128: i128;
        visit_u8: u8;
        visit_u16: u16;
        visit_u32: u32;
        visit_u64: u64;
        visit_u128: u128;
        visit_f32: f32;
        visit_f64: f64;
        visit_char: char;
        visit_str: &str;
        visit_borrowed_str: &'de str;
        visit_string: String;
        visit_bytes: &[u8];
        visit_borrowed_bytes: &'de [u8];
        visit_byte_buf: Vec<u8>;
    }

    fn visit_none<E: de::Error>(self) -> Result<V::Value, E> {
        self.0.visit_none()
    }

    fn visit_unit<E: de::Error>(self) -> Result<V::Value, E> {
        self.0.visit_unit()
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<V::Value, D::Error> {
        self.0.visit_some(Strict(deserializer))
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<V::Value, D::Error> {
        self.0.visit_newtype_struct(Strict(deserializer))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<V::Value, A::Error> {
        self.0.visit_seq(Strict(seq))
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<V::Value, A::Error> {
        self.0.visit_map(Strict(map))
    }

    fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<V::Value, A::Error> {
        self.0.visit_enum(Strict(data))
    }
}

impl<'de, A: SeqAccess<'de>> SeqAccess<'de> for Strict<A> {
    type Error = A::Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, A::Error> {
        self.0.next_element_seed(Strict(seed))
    }

    fn size_hint(&self) -> Option<usize> {
        self.0.size_hint()
    }
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for Strict<A> {
    type Error = A::Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
        self.0.next_key_seed(seed)
    }

    fn next_value_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<T::Value, A::Error> {
        self.0.next_value_seed(Strict(seed))
    }

    fn size_hint(&self) -> Option<usize> {
        self.0.size_hint()
    }
}

impl<'de, A: EnumAccess<'de>> EnumAccess<'de> for Strict<A> {
    type Error = A::Error;
    type Variant = Strict<A::Variant>;

    fn variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<(T::Value, Self::Variant), A::Error> {
        let (value, variant) = self.0.variant_seed(seed)?;
        Ok((value, Strict(variant)))
    }
}

impl<'de, A: VariantAccess<'de>> VariantAccess<'de> for Strict<A> {
    type Error = A::Error;

    fn unit_variant(self) -> Result<(), A::Error> {
        self.0.unit_variant()
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, A::Error> {
        self.0.newtype_variant_seed(Strict(seed))
    }

    fn tuple_variant<V: Visitor<'de>>(self, len: usize, visitor: V) -> Result<V::Value, A::Error> {
        self.0.tuple_variant(len, Strict(visitor))
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, A::Error> {
        self.0.struct_variant(fields, Strict(visitor))
    }
}

/// A value that is parsed and dropped.
struct Checked;

impl<'de> DeserializeSeed<'de> for Checked {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Checked {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E: de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E: de::Error>(self, _: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while seq.next_element_seed(Checked)?.is_some() {}
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while map.next_key_seed(Checked)?.is_some() {
            map.next_value_seed(Checked)?;
        }
        Ok(())
    }
}

/// A list that may be absent, in the room its items take. Read from the text
/// a list grows by doubling, where the derived read sizes it from its tree: a
/// list of one item would hold room for four.
struct Fitted<T>(PhantomData<T>);

pub(crate) fn fitted<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<T>>, D::Error> {
    deserializer.deserialize_option(Fitted(PhantomData))
}

impl<'de, T: Deserialize<'de>> Visitor<'de> for Fitted<T> {
    type Value = Option<Vec<T>>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a sequence")
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        // A replayed list names its length, the text of a body names none. The
        // hint is a claim of the source: room for a mebibyte of items at most.
        let bound = 1024 * 1024 / size_of::<T>().max(1);
        let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(bound));
        while let Some(item) = seq.next_element()? {
            if items.is_empty() {
                items.reserve_exact(1);
            }
            items.push(item);
        }
        items.shrink_to_fit();
        Ok(Some(items))
    }
}

/// Wraps the definition of an enum read by a kind key (`#[serde(tag = "...")]`,
/// every variant with named fields and a comma after its last field), derives
/// its `Deserialize` (the enum's derive list leaves it out, so no other reader
/// of the enum can stand beside serde's) and writes its [`ByKind`]: the kinds
/// as serde's own variant identifier reads them, and for each variant a struct
/// of its fields with their serde attributes, read by serde's own derive, so
/// every field rule stays serde's. An attribute of a field that is not serde's
/// is left out of that struct; a field written `Option<Vec<T>>` with no serde
/// attribute is read by [`fitted`]. What the kinds and the fields would not
/// read as the derive of the enum does is refused at compile time: the enum
/// and its variants may carry the attributes of `@known` alone (on a variant a
/// doc comment and `#[serde(rename = "...")]`); an attribute of a field may
/// neither stand behind `cfg_attr` nor come from another macro as a fragment,
/// since either could hide a serde attribute; and a field may not stand
/// behind `cfg`, which that struct would leave out. In a field's type the
/// enum goes by its name: `Self` there would mean that struct.
macro_rules! by_kind {
    (@key #[serde(tag = $key:literal)] $($more:tt)*) => { $key };
    (@key #[$($other:tt)*] $($more:tt)*) => { $crate::kind_read::by_kind!(@key $($more)*) };
    (@known #[doc = $doc:literal] $($more:tt)*) => { $crate::kind_read::by_kind!(@known $($more)*); };
    (@known #[derive($($derive:tt)*)] $($more:tt)*) => { $crate::kind_read::by_kind!(@known $($more)*); };
    (@known #[serde_with::skip_serializing_none] $($more:tt)*) => { $crate::kind_read::by_kind!(@known $($more)*); };
    (@known #[serde(tag = $key:literal)] $($more:tt)*) => { $crate::kind_read::by_kind!(@known $($more)*); };
    (@known #[serde(rename = $name:literal)] $($more:tt)*) => { $crate::kind_read::by_kind!(@known $($more)*); };
    (@known) => {};
    (@f $ctx:tt [$($done:tt)*] [$($names:ident)*] [$($keep:tt)*] #[serde $s:tt] $($more:tt)*) => {
        $crate::kind_read::by_kind!(@f $ctx [$($done)*] [$($names)*] [$($keep)* #[serde $s]] $($more)*)
    };
    (@f $ctx:tt [$($done:tt)*] [$($names:ident)*] [$($keep:tt)*] #[cfg_attr $($no:tt)*] $($more:tt)*) => {
        compile_error!("by_kind! does not read a field attribute behind cfg_attr")
    };
    (@f $ctx:tt [$($done:tt)*] [$($names:ident)*] [$($keep:tt)*] #[cfg $($no:tt)*] $($more:tt)*) => {
        compile_error!("by_kind! does not read a field behind cfg")
    };
    (@f $ctx:tt [$($done:tt)*] [$($names:ident)*] [$($keep:tt)*] #[$attr:ident $($args:tt)*] $($more:tt)*) => {
        $crate::kind_read::by_kind!(@f $ctx [$($done)*] [$($names)*] [$($keep)*] $($more)*)
    };
    (@f $ctx:tt [$($done:tt)*] [$($names:ident)*] [$($keep:tt)*] #[$($no:tt)*] $($more:tt)*) => {
        compile_error!("by_kind! does not read a field attribute whose name it cannot see")
    };
    (@f $ctx:tt [$($done:tt)*] [$($names:ident)*] [] $field:ident : Option<Vec<$item:ty>>, $($more:tt)*) => {
        $crate::kind_read::by_kind!(@f $ctx
            [$($done)* #[serde(default, deserialize_with = "crate::kind_read::fitted")] $field: Option<Vec<$item>>,]
            [$($names)* $field] [] $($more)*)
    };
    (@f $ctx:tt [$($done:tt)*] [$($names:ident)*] [$($keep:tt)*] $field:ident : $fty:ty, $($more:tt)*) => {
        $crate::kind_read::by_kind!(@f $ctx [$($done)* $($keep)* $field: $fty,] [$($names)* $field] [] $($more)*)
    };
    (@f ($name:ident $variant:ident $rest:ident) [$($done:tt)*] [$($names:ident)*] []) => {{
        #[derive(serde::Deserialize)]
        struct __ByKindFields {
            $($done)*
        }
        let fields = <__ByKindFields as serde::Deserialize>::deserialize($rest)?;
        Ok($name::$variant { $($names: fields.$names),* })
    }};
    (
        $(#[$($emeta:tt)*])*
        $vis:vis enum $name:ident {
            $( $(#[$($vmeta:tt)*])* $variant:ident { $($body:tt)* } ),* $(,)?
        }
    ) => {
        #[derive(serde::Deserialize)]
        $(#[$($emeta)*])*
        $vis enum $name {
            $( $(#[$($vmeta)*])* $variant { $($body)* } ),*
        }

        const _: () = {
            $crate::kind_read::by_kind!(@known $(#[$($emeta)*])*);
            $( $crate::kind_read::by_kind!(@known $(#[$($vmeta)*])*); )*

            // A field's type is named again inside `of_kind`, where these items
            // are in scope too: they carry names no type of a caller has.
            #[derive(serde::Deserialize)]
            #[serde(variant_identifier)]
            pub(crate) enum __ByKindKind {
                $( $(#[$($vmeta)*])* $variant ),*
            }

            impl $crate::kind_read::ByKind for $name {
                const KEY: &'static str = $crate::kind_read::by_kind!(@key $(#[$($emeta)*])*);
                const EXPECTING: &'static str = concat!("internally tagged enum ", stringify!($name));
                type Kind = __ByKindKind;

                fn of_kind<'de, __D: serde::Deserializer<'de>>(
                    kind: __ByKindKind,
                    rest: __D,
                ) -> ::std::result::Result<Self, __D::Error> {
                    match kind {
                        $( __ByKindKind::$variant => $crate::kind_read::by_kind!(@f ($name $variant rest) [] [] [] $($body)*), )*
                    }
                }
            }
        };
    };
}
pub(crate) use by_kind;

#[cfg(all(test, feature = "axum"))]
mod tests {
    use super::direct;
    use crate::chat::ChatCompletionRequest;

    /// Every message shape, read by both roads: the same value or a refusal by
    /// both. The texts of two refusals may differ in place: the parse that
    /// words a refusal for a client runs the derived road.
    #[test]
    fn a_message_is_read_as_the_derive_reads_it() {
        let good = [
            r#"{"role":"user","content":[{"type":"nothing"}]}"#,
            r#"{"role":"system","content":"x","tools":7}"#,
            r#"{"role":"\u0075ser","content":"an escaped role"}"#,
            r#"{ "role" : "user" , "content" : "spaces" }"#,
            r#"{"role":"user","content":"x","zz":1,"zz":2}"#,
            r#"{"role":"user","content":"x","tools":[{"type":"function"}]}"#,
            r#"{"role":"assistant","content":"a","reasoning":"why","reasoning_content":"why"}"#,
            r#"{"role":"assistant","content":null,"audio":{"id":"a1"},"function_call":{"name":"f","arguments":"{}"},"tools":{"k":1}}"#,
            r#"{"role":"developer","content":"rules","name":"d","zz":null}"#,
            r#"{"role":"tool","content":[{"type":"text","text":"part"}],"tool_call_id":"c1","extra":true}"#,
            r#"{"role":"system","content":"be brief"}"#,
            r#"{"role":"system"}"#,
            r#"{"role":"system","name":null,"content":"x","partial":true,"zz":[1,{"a":2}]}"#,
            r#"{"role":"user","content":"hello \"there\"\n"}"#,
            r#"{"content":"role last","role":"user"}"#,
            r#"{"tools":7,"role":"system","content":"x"}"#,
            r#"{"tools":[{"type":"function"}],"role":"user","content":"x"}"#,
            r#"{"reasoning":"why","reasoning_content":"why","role":"assistant","content":"a"}"#,
            r#"{"audio":{"id":"a1"},"content":null,"role":"assistant"}"#,
            r#"{"name":"n","role":"user","content":"role in the middle"}"#,
            r#"{"\u0072ole":"user","content":"an escaped key"}"#,
            r#"{"content":"x","\u006eame":"n","role":"user"}"#,
            r#"{"content":[{"type":"text","text":"a part"}],"name":"n","role":"user"}"#,
            r#"{"content":null,"refusal":null,"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}"#,
            r#"{"zz":{"a":[1,{"b":null}]},"content":"x","role":"tool","tool_call_id":"c1"}"#,
            r#"{"partial":true,"zz":1.5,"content":"x","role":"assistant"}"#,
            r#"{"function_call":{"name":"f","arguments":"{}"},"content":null,"role":"assistant"}"#,
            r#"{"content":"a \"quoted\" text","refusal":"no","role":"assistant"}"#,
            r#"{"role":"user","content":[{"type":"text","text":"a part"}],"name":"n"}"#,
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}],"reasoning_content":"why","other":1.5}"#,
            r#"{"role":"assistant","content":"answer","refusal":null,"name":"a"}"#,
            r#"{"role":"assistant"}"#,
            r#"{"role":"tool","content":"result","tool_call_id":"c1"}"#,
            r#"{"tool_call_id":"c1","content":"result","role":"tool"}"#,
            r#"{"role":"function","content":null,"name":"f"}"#,
            r#"{"role":"developer","content":"rules"}"#,
            r#"{"role":"root","content":"r","name":null}"#,
            r#"{"role":"assistant","tool_calls":null}"#,
            r#"{"tool_calls":[],"role":"assistant"}"#,
            r#"{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"c2","type":"function","function":{"name":"g","arguments":"{}"},"zz":1}]}"#,
            r#"{"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"c2","type":"function","function":{"name":"g","arguments":"{}"}}],"content":null,"role":"assistant"}"#,
            r#"["tool","result","c1"]"#,
            r#"["function","text","f"]"#,
            r#"["root","r","n"]"#,
        ];
        let bad = [
            r#"{"role":"nobody","content":"x"}"#,
            r#"{"content":"x","role":"nobody"}"#,
            r#"{"content":"x"}"#,
            r"{}",
            r#"{"role":1,"content":"x"}"#,
            r#"{"role":null,"content":"x"}"#,
            r#"{"role":"user"}"#,
            r#"{"role":"user","content":5}"#,
            r#"{"role":"user","content":"a","content":"b"}"#,
            r#"{"role":"user","role":"user","content":"x"}"#,
            r#"{"role":"user","content":"x","role":"user"}"#,
            r#"{"content":"x","role":"user","role":"user"}"#,
            r#"{"content":"a","content":"b","role":"user"}"#,
            r#"{"content":5,"role":"user"}"#,
            r#"{"content":"x" "role":"user"}"#,
            r#"{"tool_calls":"none","role":"assistant"}"#,
            r#"{"audio":{},"content":null,"role":"assistant"}"#,
            r#"{"name":5,"content":"x","role":"user"}"#,
            r#"{"role":"tool","content":"x"}"#,
            r#"{"role":"tool","content":"x","tool_call_id":7}"#,
            r#"{"role":"function","content":"x"}"#,
            r#"{"role":"assistant","tool_calls":"none"}"#,
            r#"{"role":"assistant","audio":{}}"#,
            r#"{"role":"user","content":"x""#,
            r#""user""#,
            r"7",
            r#"["nobody","x"]"#,
            r#"["user","x"]"#,
            r"[]",
            r"null",
            r#"{"role":"assistant","reasoning":"a","reasoning_content":"b"}"#,
            r#"{"reasoning":"a","reasoning_content":"b","role":"assistant"}"#,
            r#"{"role":"tool","content":"a","content":"b","tool_call_id":"c"}"#,
            r#"{"role":"user","content":"x","name":5}"#,
            r#"{"role":"assistant","tool_calls":[{}]}"#,
            r#"["tool","result"]"#,
            r#"["tool","result","c1","more"]"#,
            r#"{"role":"user","content":"x",}"#,
            r#"{"role":"user" "content":"x"}"#,
            r#"{"role":"us\u0000er","content":"x"}"#,
            r#"{"role":"tool","content":"x","tool_call_id":"c1","zz":"\udc00"}"#,
            r#"{"role":"assistant","tool_calls":{"a":1}}"#,
            r#"{"role":"assistant","tool_calls":[],"tool_calls":[]}"#,
            r#"{"tool_calls":5,"role":"assistant"}"#,
            r#"{"tool_calls":[7],"role":"assistant"}"#,
        ];
        let read = |message: &str, direct_road: bool| {
            let body = format!(
                r#"{{"model":"m","messages":[{message},{{"role":"user","content":"end"}}]}}"#
            );
            let parse = || serde_json::from_str::<ChatCompletionRequest>(&body);
            let parsed = if direct_road {
                direct::<ChatCompletionRequest, _>(parse)
            } else {
                parse()
            };
            parsed
                .map(|request| format!("{:?}", request.messages))
                .map_err(|e| e.to_string())
        };
        let mut misplaced = Vec::new();
        for message in good {
            let (direct, derived) = (read(message, true), read(message, false));
            if derived.is_err() {
                misplaced.push(format!("the derive refuses {message}: {derived:?}"));
            }
            assert_eq!(direct, derived, "{message}");
        }
        for message in bad {
            let (direct, derived) = (read(message, true), read(message, false));
            if derived.is_ok() {
                misplaced.push(format!("the derive accepts {message}"));
            }
            assert_eq!(direct.is_ok(), derived.is_ok(), "{message}");
        }
        assert!(misplaced.is_empty(), "in the wrong list: {misplaced:#?}");
    }

    crate::kind_read::by_kind! {
    #[derive(Debug, PartialEq)]
    #[serde(tag = "kind")]
    enum Wrapped {
        #[serde(rename = "numbers")]
        Numbers {
            #[serde{alias = "ids"}]
            list: Vec<u32>,
        },
    }
    }

    /// A type other than a chat message is read by its kind. Small numbers held
    /// before the kind key reach their field, and a serde attribute reaches the
    /// struct of the fields whatever its brackets: serde reads braces too.
    #[test]
    fn a_wrapped_type_is_read_by_its_kind() {
        use serde::de::DeserializeSeed;

        let body = r#"{"ids":[1,2],"kind":"numbers"}"#;
        let read: Result<Wrapped, _> = super::Item(&mut Vec::new(), std::marker::PhantomData)
            .deserialize(&mut serde_json::Deserializer::from_str(body));
        assert_eq!(read.ok(), Some(Wrapped::Numbers { list: vec![1, 2] }));
    }
}
