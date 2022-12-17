# sharpy

Ptychography reconstruction

## Continous Integration

Github Actions is set up to test numpy operations under sharpy/tests. Manually test using:
```console
$ pytest
```

Manually test cupy operations in /tests/cupy_check.py using:
```console
$ pytest sharpy/tests/cupy_check.py
```
