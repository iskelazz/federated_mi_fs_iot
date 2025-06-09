import time, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class Timer:
    def __init__(self, tag):
        self.tag = tag
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.t0
        #logging.info(f'[TIME] {self.tag}: {elapsed:.3f}s')
        self.elapsed = elapsed