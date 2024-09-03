import pathlib
import pickle
import typing


def pickle_save(obj: typing.Any, abs_path: str | pathlib.Path) -> None:
    """ Saves the given object to the given absolute path.

    Simple wrapper around the `pickle` package.

    Args:
        obj (Any): Object to be saved.
        abs_path (str | Path): Absolute path of the saving file. If the given path
            doesn't end with the suffix ".pkl", it will be automatically
            added.
    """
    p = pathlib.Path(abs_path)
    if not p.suffixes:
        p = pathlib.Path(str(abs_path) + ".pkl")
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(str(p), "wb") as out_file:
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def pickle_load(abs_path: str | pathlib.Path) -> typing.Any:
    """ Loads an object from the given absolute path.

    Simple wrapper around the `pickle` package.

    Args:
        abs_path (str | Path): Absolute path of the saved ".pkl" file. If the given
            path doesn't end with the suffix ".pkl", it will be automatically
            added.

    Returns:
        The loaded object.
    """
    p = pathlib.Path(abs_path)
    if not p.suffixes:
        p = pathlib.Path(str(abs_path) + ".pkl")

    with open(p, "rb") as in_file:
        return pickle.load(in_file)
