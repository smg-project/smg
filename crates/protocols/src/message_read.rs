//! The read of a chat request's `messages`. Inside [`direct`] a message is
//! read straight into the variant its `role` names; a value that stands before
//! `role` is kept as it was parsed until the role is known. A message that is
//! no object and every read outside [`direct`] take the derived read of
//! [`ChatMessage`], which first copies the whole message into a generic tree.
//! Both read a body into the same value and refuse the same bodies: a value
//! the variant ignores is parsed before it is dropped, as the derived read
//! parses it. Every refusal is worded by the derived read.

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

use crate::{
    chat::{AssistantReasoning, ChatAssistantAudio, ChatMessage, MessageContent},
    common::{FunctionCallResponse, ToolCall},
    ext::kimi::{KimiAssistantExt, KimiDeveloperExt, KimiSystemExt, KimiUserExt},
};

thread_local! {
    static DIRECT: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f`, the parse of a body of type `T`, with the direct read of a
/// message on when `T` is the chat request itself. The caller parses again
/// outside when `f` refuses a body, so that the refusal names the path and the
/// place of the error as the derived read does. A chat request nested in
/// another body type may be parsed where a refusal is not retried, so it keeps
/// the derived read; a type name that does not match only turns the switch off.
#[cfg(feature = "axum")]
pub(crate) fn direct<T, R>(f: impl FnOnce() -> R) -> R {
    /// Puts the switch back when the direct read ends, also by a panic.
    struct Restore(bool);

    impl Drop for Restore {
        fn drop(&mut self) {
            DIRECT.set(self.0);
        }
    }

    let on =
        std::any::type_name::<T>() == std::any::type_name::<crate::chat::ChatCompletionRequest>();
    let _restore = Restore(DIRECT.replace(on));
    f()
}

pub(crate) fn read<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<ChatMessage>, D::Error> {
    if DIRECT.get() {
        deserializer.deserialize_seq(Messages)
    } else {
        Vec::deserialize(deserializer)
    }
}

const ROLES: &[&str] = &[
    "system",
    "user",
    "assistant",
    "tool",
    "function",
    "developer",
    "root",
];

/// What stands before `role` in a message: each key and its value.
type Before<'de> = Vec<(Cow<'de, str>, Held<'de>)>;

struct Messages;

impl<'de> Visitor<'de> for Messages {
    type Value = Vec<ChatMessage>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a sequence")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut messages = Vec::new();
        let mut before = Vec::new();
        while let Some(message) = seq.next_element_seed(Message(&mut before))? {
            messages.push(message);
        }
        Ok(messages)
    }
}

struct Message<'a, 'de>(&'a mut Before<'de>);

impl<'de> DeserializeSeed<'de> for Message<'_, 'de> {
    type Value = ChatMessage;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<ChatMessage, D::Error> {
        // What the message before this one kept is dropped here.
        self.0.clear();
        deserializer.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Message<'_, 'de> {
    type Value = ChatMessage;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("internally tagged enum ChatMessage")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<ChatMessage, A::Error> {
        ChatMessage::deserialize(SeqAccessDeserializer::new(seq))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<ChatMessage, A::Error> {
        let mut key = map.next_key_seed(Key)?;
        let before: &mut [(Cow<'de, str>, Held<'de>)] = if key.as_deref() == Some("role") {
            &mut []
        } else {
            let before = self.0;
            loop {
                match key {
                    None => return Err(de::Error::missing_field("role")),
                    Some(key) if key == "role" => break,
                    Some(key) => before.push((key, map.next_value_seed(Hold)?)),
                }
                key = map.next_key_seed(Key)?;
            }
            before.as_mut_slice()
        };
        let role = map.next_value_seed(Role)?;
        let rest = MapAccessDeserializer::new(Rest {
            before,
            at: 0,
            held: false,
            map,
        });
        Ok(match role {
            0 => {
                let m = System::deserialize(rest)?;
                ChatMessage::System {
                    content: m.content,
                    name: m.name,
                    ext: m.ext,
                }
            }
            1 => {
                let m = User::deserialize(rest)?;
                ChatMessage::User {
                    content: m.content,
                    name: m.name,
                    ext: m.ext,
                }
            }
            2 => {
                let m = Assistant::deserialize(rest)?;
                ChatMessage::Assistant {
                    content: m.content,
                    refusal: m.refusal,
                    name: m.name,
                    audio: m.audio,
                    tool_calls: m.tool_calls,
                    function_call: m.function_call,
                    reasoning: m.reasoning,
                    ext: m.ext,
                }
            }
            3 => {
                let m = Tool::deserialize(rest)?;
                ChatMessage::Tool {
                    content: m.content,
                    tool_call_id: m.tool_call_id,
                }
            }
            4 => {
                let m = Function::deserialize(rest)?;
                ChatMessage::Function {
                    content: m.content,
                    name: m.name,
                }
            }
            5 => {
                let m = Developer::deserialize(rest)?;
                ChatMessage::Developer {
                    content: m.content,
                    name: m.name,
                    ext: m.ext,
                }
            }
            _ => {
                let m = Root::deserialize(rest)?;
                ChatMessage::Root {
                    content: m.content,
                    name: m.name,
                }
            }
        })
    }
}

/// A key of a message, borrowed from the body where it holds no escape.
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

/// The value of `role`, worded as the derived variant identifier words it.
struct Role;

impl<'de> DeserializeSeed<'de> for Role {
    type Value = usize;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<usize, D::Error> {
        deserializer.deserialize_identifier(self)
    }
}

impl Visitor<'_> for Role {
    type Value = usize;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("variant identifier")
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<usize, E> {
        usize::try_from(v)
            .ok()
            .filter(|index| *index < ROLES.len())
            .ok_or_else(|| {
                E::invalid_value(de::Unexpected::Unsigned(v), &"variant index 0 <= i < 7")
            })
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<usize, E> {
        ROLES
            .iter()
            .position(|role| *role == v)
            .ok_or_else(|| E::unknown_variant(v, ROLES))
    }

    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<usize, E> {
        let v = String::from_utf8_lossy(v);
        self.visit_str(&v)
    }
}

/// The keys of a message but `role`: what stood before it, then what follows.
/// A second `role` is refused as the derived read refuses it.
struct Rest<'a, 'de, A> {
    before: &'a mut [(Cow<'de, str>, Held<'de>)],
    at: usize,
    held: bool,
    map: A,
}

impl<'de, A: MapAccess<'de>> MapAccess<'de> for Rest<'_, 'de, A> {
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
            Some(key) if key == "role" => Err(de::Error::duplicate_field("role")),
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

/// A value that stood before `role`, kept as it was parsed until the role is
/// known.
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

/// A kept value as a source for the hints the fields of a message ask for: each
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
        deserialize_i64: U64 | I64 | F64;
        deserialize_u64: U64 | I64 | F64;
        deserialize_f64: U64 | I64 | F64;
        deserialize_str: Str;
        deserialize_string: Str;
        deserialize_identifier: Str;
        deserialize_seq: Seq;
        deserialize_map: Map;
    }

    // No field of a message asks for these: they are refused, not modelled.
    refused! {
        deserialize_i8 deserialize_i16 deserialize_i32 deserialize_i128 deserialize_u8
        deserialize_u16 deserialize_u32 deserialize_u128 deserialize_f32 deserialize_char
        deserialize_bytes deserialize_byte_buf deserialize_unit
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

/// A list of tool calls in the room its calls take. Read from the text a list
/// grows by doubling, where the derived read sizes it from its tree: an
/// assistant message with one call would hold room for four.
struct ToolCalls;

fn tool_calls<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<ToolCall>>, D::Error> {
    deserializer.deserialize_option(ToolCalls)
}

impl<'de> Visitor<'de> for ToolCalls {
    type Value = Option<Vec<ToolCall>>;

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
        // hint is a claim of the source: room for a mebibyte of calls at most.
        let bound = 1024 * 1024 / size_of::<ToolCall>();
        let mut calls = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(bound));
        while let Some(call) = seq.next_element()? {
            if calls.is_empty() {
                calls.reserve_exact(1);
            }
            calls.push(call);
        }
        calls.shrink_to_fit();
        Ok(Some(calls))
    }
}

// The fields of each variant of `ChatMessage`, as its derive reads them.

#[derive(Deserialize)]
struct System {
    #[serde(default)]
    content: MessageContent,
    name: Option<String>,
    #[serde(flatten)]
    ext: KimiSystemExt,
}

#[derive(Deserialize)]
struct User {
    content: MessageContent,
    name: Option<String>,
    #[serde(flatten)]
    ext: KimiUserExt,
}

#[derive(Deserialize)]
struct Assistant {
    content: Option<MessageContent>,
    refusal: Option<String>,
    name: Option<String>,
    audio: Option<ChatAssistantAudio>,
    #[serde(default, deserialize_with = "tool_calls")]
    tool_calls: Option<Vec<ToolCall>>,
    function_call: Option<Box<FunctionCallResponse>>,
    #[serde(flatten)]
    reasoning: AssistantReasoning,
    #[serde(flatten)]
    ext: KimiAssistantExt,
}

#[derive(Deserialize)]
struct Tool {
    content: MessageContent,
    tool_call_id: String,
}

#[derive(Deserialize)]
struct Function {
    content: Option<String>,
    name: String,
}

#[derive(Deserialize)]
struct Developer {
    content: MessageContent,
    name: Option<String>,
    #[serde(flatten)]
    ext: KimiDeveloperExt,
}

#[derive(Deserialize)]
struct Root {
    content: MessageContent,
    name: Option<String>,
}

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
}
