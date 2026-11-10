"""Compatibility entry point for plain_c_rf UART evaluation."""

from .uart.evaluate import main

if __name__ == "__main__":
    main(backend="plain_c_rf")
