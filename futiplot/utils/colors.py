from types import SimpleNamespace

# define a color dictionary
futicolor = SimpleNamespace(
    dark="#03151E",
    dark1="#062230",
    dark2="#0E374B",
    light="#FFFFFF",
    light1="#6A7A83",
    pink="#EA1F96",
    pink1="#832A5E",
    purple="#6A62F8",
    blue="#00B7FF",
    blue1="#1B7CA1",
    green="#0FE6B4"
)

# define futicolor's standard gradient
futicolor.gradient = [
    futicolor.pink,
    futicolor.purple,
    futicolor.blue,
    futicolor.green
]

"""
Usage:
    from futiplot.colors import futicolor

    # Access colors
    print(futicolor.dark0)   # Output: "#03151E"
    print(futicolor.pink)    # Output: "#EA1F96"
    print(futicolor.green)   # Output: "#0FE6B4"
"""