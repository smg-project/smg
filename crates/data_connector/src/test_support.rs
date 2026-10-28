//! What the crate's unit tests share: fixtures with no home in one store.
//! Compiled for the crate's own tests only.

use std::net::{Ipv4Addr, SocketAddr};

use tokio::net::TcpSocket;

/// A loopback address nothing answers at, refused for as long as this value
/// lives (the shape of the gRPC client's `channel::testing::RefusedEndpoint`).
///
/// The socket behind it is bound and never listening: the host's own stack
/// refuses every connection at once, the way an unreachable peer's port
/// does, and the port stays taken, so no other test's `bind("127.0.0.1:0")`
/// can draw it in the meantime. A listener bound and dropped for the same
/// purpose gives its port back, and a neighbouring test's stub then answers
/// in the unreachable peer's place.
///
/// Keep the value in a named binding for the whole case: a `_` pattern or a
/// temporary drops it, and the port is free again.
#[must_use = "the address is refused only while this value lives"]
pub(crate) struct RefusedAddress {
    _reservation: TcpSocket,
    addr: SocketAddr,
}

impl RefusedAddress {
    /// Reserve a loopback port of the system's choice, and refuse it.
    pub(crate) fn reserve() -> Self {
        let socket = TcpSocket::new_v4().expect("a TCP socket");
        socket
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("a loopback port of the system's choice");
        let addr = socket.local_addr().expect("the bound address");
        Self {
            _reservation: socket,
            addr,
        }
    }

    /// The address, `127.0.0.1:<port>`.
    pub(crate) fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The port alone, for a URL built around it.
    pub(crate) fn port(&self) -> u16 {
        self.addr.port()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::ErrorKind,
        net::{TcpListener, TcpStream},
    };

    use super::RefusedAddress;

    /// A connection to the address is refused at once, and no other bind
    /// takes the port while the value lives.
    #[test]
    fn the_address_refuses_connections_and_keeps_its_port() {
        let refused = RefusedAddress::reserve();
        let error = TcpStream::connect(refused.addr()).expect_err("nothing answers there");
        assert_eq!(error.kind(), ErrorKind::ConnectionRefused, "{error}");
        let taken = TcpListener::bind(refused.addr()).expect_err("the port is held");
        assert_eq!(taken.kind(), ErrorKind::AddrInUse, "{taken}");
        assert_eq!(refused.addr().port(), refused.port());
    }
}
