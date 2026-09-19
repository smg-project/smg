use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};

/// Read at most `cap` bytes and report whether the body exceeds that limit.
/// Return as soon as excess data arrives. Preserve errors before that point.
pub async fn read_body_capped<S, E>(stream: S, cap: usize) -> Result<(Bytes, bool), E>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    futures::pin_mut!(stream);
    let mut body = BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let remaining = cap.saturating_sub(body.len());
        if chunk.len() > remaining {
            body.extend_from_slice(&chunk[..remaining]);
            return Ok((body.freeze(), true));
        }
        body.extend_from_slice(&chunk);
    }
    Ok((body.freeze(), false))
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures::{executor::block_on, stream, StreamExt};

    use super::read_body_capped;

    type Case = (&'static [&'static [u8]], usize, &'static [u8], bool);

    #[test]
    fn retains_only_the_prefix_and_reports_actual_overflow() {
        let cases: &[Case] = &[
            (&[], 0, b"", false),
            (&[b""], 0, b"", false),
            (&[b"x"], 0, b"", true),
            (&[b"ab", b"cd"], 8, b"abcd", false),
            (&[b"ab", b"cd"], 4, b"abcd", false),
            (&[b"ab", b"cd", b""], 4, b"abcd", false),
            (&[b"ab", b"cd", b"e"], 4, b"abcd", true),
            (&[b"abcdef"], 4, b"abcd", true),
            (&[b"a"], usize::MAX, b"a", false),
        ];
        for &(chunks, cap, expected, truncated) in cases {
            let input = stream::iter(
                chunks
                    .iter()
                    .map(|chunk| Ok::<_, &'static str>(Bytes::copy_from_slice(chunk))),
            );
            let (body, actual) = block_on(read_body_capped(input, cap)).unwrap();
            assert_eq!(body.as_ref(), expected);
            assert_eq!(actual, truncated);
            assert!(body.len() <= cap);
        }
    }

    #[test]
    fn returns_a_read_error_before_or_at_the_cap() {
        for cap in [3, 4] {
            let input = stream::iter([Ok(Bytes::from_static(b"abc")), Err("body read failed")]);
            assert_eq!(
                block_on(read_body_capped(input, cap)),
                Err("body read failed")
            );
        }
    }

    #[test]
    fn does_not_poll_the_tail_after_overflow() {
        let mut tail_polled = false;
        let tail = stream::poll_fn(|_| {
            tail_polled = true;
            std::task::Poll::Ready(Some(Err::<Bytes, _>("tail was read")))
        });
        let input = stream::iter([Ok(Bytes::from_static(b"abcde"))]).chain(tail);
        let (body, truncated) = block_on(read_body_capped(input, 4)).unwrap();
        assert_eq!(body.as_ref(), b"abcd");
        assert!(truncated);
        assert!(!tail_polled);
    }

    #[test]
    fn accepts_a_stream_that_is_not_unpin() {
        let input = stream::once(async { Ok::<_, &'static str>(Bytes::from_static(b"ok")) });
        let (body, truncated) = block_on(read_body_capped(input, 2)).unwrap();
        assert_eq!(body.as_ref(), b"ok");
        assert!(!truncated);
    }
}
