"""Hello World Ray Job Example."""
import ray


@ray.remote
def hello_world():
    """A simple remote function that returns 'hello world'."""
    return "hello world"


ray.init()
print(ray.get(hello_world.remote()))
