class LiftError(ValueError):

    def __init__(self, value, target_class):
        msg = f"Cannot lift {value} of type {type(value)} to {target_class}"
        super().__init__(msg)
        self.value = value
        self.target_class = target_class
