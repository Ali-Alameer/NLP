from functools import wraps
import threading
import time
import polars as pl


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float = 1.0):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.rate = float(rate_per_sec)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = now - self.updated_at
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated_at = now

    def acquire(self, tokens: float = 1.0) -> None:
        with self.lock:
            requested_tokens = float(tokens)
            if requested_tokens <= 0:
                raise ValueError("tokens must be > 0")
            if requested_tokens > self.capacity:
                raise ValueError(
                    f"\033[91mtokens ({requested_tokens}) cannot exceed bucket capacity ({self.capacity})\033[0m"
                )

            print(
                f"\033[94mAcquiring {requested_tokens:.2f} tokens from the bucket.\033[0m"
            )
            while True:
                now = time.monotonic()
                self._refill(now)

                if self.tokens >= requested_tokens:
                    self.tokens -= requested_tokens
                    print(
                        f"\033[94mAcquired {requested_tokens:.2f} tokens, {self.tokens:.2f} remaining in the bucket.\033[0m"
                    )
                    return

                needed = requested_tokens - self.tokens
                wait_s = needed / self.rate

                time.sleep(wait_s)


def retry_on_exception(max_retries):
    """Decorator to retry a function on exception with exponential backoff.

    Args:
        max_retries (int): Maximum retry attempts.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    wait_time = 2**retries
                    print(
                        f"\033[93mException occurred: {e}. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})\033[0m"
                    )
                    time.sleep(wait_time)
            raise Exception(
                f"\033[91mFunction {func.__name__} failed after {max_retries} retries.\033[0m"
            )

        return wrapper

    return decorator


def time_it(func):
    """Decorator to measure execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(
            f"\033[92mExecution time for {func.__name__}: {elapsed:.2f} seconds\033[0m"
        )
        return result

    return wrapper


def upsert_jsonl_dataset(
    dataset: list[dict], file_path: str, deduplicate: bool = True
) -> None:
    """Upsert a dataset to a jsonl file.

    Args:
        dataset (list[dict]): List of product data dictionaries to be saved.
        file_path (str): Path to jsonl file where the dataset should be saved.
        deduplicate (bool, optional): Whether to deduplicate the dataset. Defaults to True.
    """
    new_df = pl.DataFrame(dataset)
    try:
        existing_df = pl.read_ndjson(file_path, infer_schema_length=None)
        if deduplicate:
            combined_df = pl.concat(
                [existing_df, new_df], how="diagonal_relaxed").unique()
        else:
            combined_df = pl.concat(
                [existing_df, new_df], how="diagonal_relaxed")
    except FileNotFoundError:
        print(
            f"\033[93mFile {file_path} not found. Creating new dataset.\033[0m")
        combined_df = new_df
    combined_df.write_ndjson(file_path)
    print(
        f"\033[92mDataset upserted to {file_path} with {len(combined_df)} total records.\033[0m"
    )
