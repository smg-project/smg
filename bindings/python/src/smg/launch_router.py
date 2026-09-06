from __future__ import annotations

import argparse
import logging
import sys

import setproctitle

from smg.router_args import RouterArgs
from smg.smg_rs import print_banner

logger = logging.getLogger("router")

try:
    from smg.router import Router
except ImportError:
    Router = None  # type: ignore[assignment,misc]
    logger.warning("Rust Router is not installed")


def launch_router(args: argparse.Namespace | RouterArgs) -> None:
    """
    Launch the SMG router with the configuration from parsed arguments.

    Args:
        args: Namespace object containing router configuration
            Can be either raw argparse.Namespace or converted RouterArgs

    Returns:
        None. Raises on failure.
    """
    setproctitle.setproctitle("smg::router")
    try:
        # Convert to RouterArgs if needed
        if not isinstance(args, RouterArgs):
            router_args = RouterArgs.from_cli_args(args)
        else:
            router_args = args

        if Router is None:
            raise RuntimeError("Rust Router is not installed")

        if router_args.service_discovery and not router_args.enable_igw:
            logger.info("IGW mode automatically enabled because service discovery is turned on")
            router_args.enable_igw = True

        router_args._validate_router_args()

        # Determine mode for banner
        if getattr(router_args, "enable_igw", False):
            mode = "IGW"
        elif getattr(router_args, "pd_disaggregation", False):
            mode = "PD Disaggregated"
        else:
            mode = "Regular"
        print_banner(router_args.host, router_args.port, mode)

        router = Router.from_args(router_args)
        router.start()

    except Exception as e:
        logger.error(f"Error starting router: {e}")
        raise e


class CustomHelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Custom formatter that preserves both description formatting and shows defaults"""

    pass


def parse_router_args(args: list[str]) -> RouterArgs:
    """Parse command line arguments and return RouterArgs instance."""
    parser = argparse.ArgumentParser(
        description="""SMG Router - High-performance request distribution across worker nodes

Usage:
This launcher enables starting a router with individual worker instances. It is useful for
multi-node setups or when you want to start workers and router separately.

Examples:
  # Regular mode
  python -m smg.launch_router --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode with same policy for both
  python -m smg.launch_router --pd-disaggregation \\
    --prefill http://prefill1:8000 9000 --prefill http://prefill2:8000 \\
    --decode http://decode1:8001 --decode http://decode2:8001 \\
    --policy cache_aware

  # PD mode with optional bootstrap ports
  python -m smg.launch_router --pd-disaggregation \\
    --prefill http://prefill1:8000 9000 \\    # With bootstrap port
    --prefill http://prefill2:8000 none \\    # Explicitly no bootstrap port
    --prefill http://prefill3:8000 \\         # Defaults to no bootstrap port
    --decode http://decode1:8001 --decode http://decode2:8001

  # PD mode with different policies for prefill and decode
  python -m smg.launch_router --pd-disaggregation \\
    --prefill http://prefill1:8000 --prefill http://prefill2:8000 \\
    --decode http://decode1:8001 --decode http://decode2:8001 \\
    --prefill-policy cache_aware --decode-policy power_of_two

  # EPD disaggregated mode
  python -m smg.launch_router --epd-disaggregation \\
    --encode http://encode1:8000 9000 \\
    --prefill http://prefill1:8000 9001 \\
    --decode http://decode1:8001 \\
    --encode-policy consistent_hashing

    """,
        formatter_class=CustomHelpFormatter,
    )

    RouterArgs.add_cli_args(parser, use_router_prefix=False)
    return RouterArgs.from_cli_args(parser.parse_args(args), use_router_prefix=False)


def main() -> None:
    router_args = parse_router_args(sys.argv[1:])
    launch_router(router_args)


if __name__ == "__main__":
    main()
