def protect(*protected):
    """Returns a metaclass that protects all attributes given as strings"""

    class Protect(type):
        has_base = False

        def __new__(mcs, name, bases, attrs):
            if mcs.has_base:
                for attribute in attrs:
                    if attribute in protected:
                        raise AttributeError(
                            f'Overriding of attribute "{attribute}" is not allowed.'
                        )
            mcs.has_base = True
            klass = super().__new__(mcs, name, bases, attrs)
            return klass

    return Protect
