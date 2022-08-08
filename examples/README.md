![Al](../al.svg) Aluminum Examples
==================================

These are simple examples of how to use Aluminum.

Current examples:
* `hello_world`: Initialize Aluminum and have each process print its rank.
* `allreduce`: Allreduce a buffer of data and print it afterward.

## Building

You can build the examples as follows.
This assumes Aluminum has already been installed.

```
mkdir build
cd build
cmake ..
make
```

If CMake cannot find the Aluminum library automatically, pass `-D Aluminum_DIR=/path/to/Aluminum`.

You can edit the example codes to change certain parameters, like the Aluminum backend used.
