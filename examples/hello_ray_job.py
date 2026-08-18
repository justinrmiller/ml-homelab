"""Hello World Ray Job Example."""

import ray


def hello_world() -> str:
    """Return a greeting. Kept undecorated so it is callable in tests."""
    return "hello world"


# The remote handle Ray schedules on the cluster.
hello_world_remote = ray.remote(hello_world)


def main() -> None:
    """Connect to Ray and print the result of the remote call."""
    ray.init()
    print(ray.get(hello_world_remote.remote()))


if __name__ == "__main__":
    main()
