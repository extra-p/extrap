from tqdm import tqdm


class ProgressBar(tqdm):
    def __init__(self, **kwargs):
        if 'iterable' not in kwargs:
            kwargs['total'] = 0
        if 'bar_format' not in kwargs:
            kwargs['bar_format'] = '{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]'
        super().__init__(**kwargs)

    def step(self, s='', refresh=True):
        super().set_postfix_str(s, refresh)

    def __call__(self, iterable, length=None, scale=1):
        if length:
            self.total += length * scale
        else:
            try:
                self.total += len(iterable) * scale
            except (TypeError, AttributeError):
                pass

        for obj in iterable:
            yield obj
            self.update(scale)


class DummyProgressBar(ProgressBar):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, total=0, disable=True)

    def update(self, n=1):
        pass

    def clear(self, nolock=False):
        pass

    def refresh(self, nolock=False, lock_args=None):
        pass

    def step(self, s='', refresh=True):
        pass

    def __call__(self, iterable, length=None, scale=1):
        return iterable

    def unpause(self):
        pass

    def reset(self, total=None):
        pass

    def display(self, msg=None, pos=None):
        return True

    def close(self):
        pass


DUMMY_PROGRESS = DummyProgressBar()
