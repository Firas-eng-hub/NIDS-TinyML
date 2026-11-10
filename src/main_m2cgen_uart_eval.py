"""Compatibility entry point for m2cgen UART evaluation."""

from .uart.evaluate import main

if __name__ == "__main__":
    main(backend="m2cgen")
