class MatplotlibTorchImportWorkaround:
    """
    Matplotlib very nicely check if the given input is a torch tensor, without
    actually importing torch, via sys.modules.get("torch").Tensor, and checking for AttributeError
    (i.e. relying on get(...) returning None and accessing its attribute).

    However, it breaks this lazy import workflow as torch is *imported* as lazy module, and accessing
    it's attribute forces it to load torch.

    We will manually raise the attribute error if we detected that it's matplotlib that's trying to
    load torch.
    """

    def __enter__(self):
        import inspect

        for line in inspect.stack():
            if (
                "matplotlib/cbook" in line.filename
                and line.function == "_is_torch_array"
            ):
                raise AttributeError("torch is not imported")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
