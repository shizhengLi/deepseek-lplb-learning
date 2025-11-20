import importlib.util
import os

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


default_cache_dir = os.getenv('DEFAULT_CACHE_DIR', os.path.expanduser('~/.lplb'))
deep_ep_path = importlib.util.find_spec('deep_ep_cpp')
if deep_ep_path:
    deep_ep_path = deep_ep_path.origin
    nvshmem_dir = os.environ['NVSHMEM_DIR']

setuptools.setup(
    ext_modules=[
        CUDAExtension(
            name='lplb._cpp',
            sources=['csrc/plugin.cpp'],
            libraries=[
                'nvrtc',
                'nvJitLink',
                'cuda',
            ],
            extra_link_args=[
                *(
                    [deep_ep_path, '-Wl,-rpath,' + os.path.dirname(deep_ep_path)]
                    if deep_ep_path
                    else []
                ),
                '-L/usr/local/cuda/lib64/stubs',
            ],
            include_dirs=[f'{nvshmem_dir}/include'] if deep_ep_path else [],
            extra_compile_args={
                'cxx': [
                    f'-DDEFAULT_CACHE_DIR="{default_cache_dir}"',
                    *(
                        [
                            '-DUSE_NVSHMEM',
                            f'-DNVSHMEM_DIR="{nvshmem_dir}"',
                            f'-DDEEP_EP_SO="{deep_ep_path}"',
                        ]
                        if deep_ep_path
                        else []
                    ),
                ]
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
