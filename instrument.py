import time

import pyvisa


class RequestHandler:
    """A wrapper around pyvisa that limits the rate of requests.

    Parameters
    ----------
    resource_name (str)
        The name of the resource.
    interval_ms (int)
        The interval in milliseconds between requests.
    **kwargs
        Additional keyword arguments to be passed to the resource.

    """

    def __init__(self, resource_name: str, interval_ms: int = 50, **kwargs):
        self.resource_name = resource_name
        self.interval_ms = interval_ms
        self._resource_kwargs = kwargs

    def open(self):
        """Open the pyvisa resource."""
        self.inst = pyvisa.ResourceManager().open_resource(
            self.resource_name, **self._resource_kwargs
        )
        self._last_update = time.perf_counter_ns()

    def wait_time(self):
        """Wait until the interval between requests has passed."""
        if self.interval_ms == 0:
            return
        while (time.perf_counter_ns() - self._last_update) <= self.interval_ms * 1e3:
            time.sleep(1e-4)

    def write(self, *args, **kwargs):
        self.wait_time()
        res = self.inst.write(*args, **kwargs)
        self._last_update = time.perf_counter_ns()
        return res

    def query(self, *args, **kwargs):
        self.wait_time()
        res = self.inst.query(*args, **kwargs)
        self._last_update = time.perf_counter_ns()
        return res

    def read(self, *args, **kwargs):
        """Read data from the resource.

        This is not very likely to be used. It may cause problems due to the wait time.
        Use `query` instead.
        """
        self.wait_time()
        res = self.inst.read(*args, **kwargs)
        self._last_update = time.perf_counter_ns()
        return res

    def close(self):
        self.inst.close()
